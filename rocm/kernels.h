/* Qwen3-ASR HIP kernels. Wave32 GEMV and norm/RoPE design adapted from
 * ds4.c/rocm/ds4_rocm_common.cuh and ds4_rocm_norm_rope.cuh (LICENSE.ds4).
 * Accumulations, normalization, attention and residuals remain float32. */
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

__device__ inline float wave_sum(float v) {
    for (int d=16; d; d>>=1) v += __shfl_down(v,d,32);
    return __shfl(v,0,32);
}
__device__ inline float wave_max(float v) {
    for (int d=16; d; d>>=1) v = fmaxf(v,__shfl_down(v,d,32));
    return __shfl(v,0,32);
}
template<bool MAX=false> __device__ float block_reduce(float v) {
    __shared__ float sums[32];
    int lane=threadIdx.x&31, wave=threadIdx.x>>5;
    v=MAX?wave_max(v):wave_sum(v);
    if (!lane) sums[wave]=v;
    __syncthreads();
    v=threadIdx.x<(blockDim.x/32)?sums[lane]:(MAX?-INFINITY:0.f);
    if (!wave) v=MAX?wave_max(v):wave_sum(v);
    if (!threadIdx.x) sums[0]=v;
    __syncthreads();
    float result=sums[0];
    __syncthreads();
    return result;
}
__device__ inline float gelu(float x) {
    return .5f*x*(1.f+tanhf(.7978845608028654f*(x+.044715f*x*x*x)));
}
__global__ void to_half(half *out,const float *x,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i]=__float2half(x[i]);
}
__global__ void bf16_to_half(half *out,const uint16_t *x,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i]=__float2half(__uint_as_float((unsigned)x[i]<<16));
}
__global__ void quantize_q8(int8_t *out,float *scale,const half *x,size_t blocks) {
    size_t b=(size_t)blockIdx.x*8+(threadIdx.x>>5);
    int lane=threadIdx.x&31;
    if (b>=blocks) return;
    float v=__half2float(x[b*32+lane]);
    float a=wave_max(fabsf(v)), s=a/127.f;
    if (!lane) scale[b]=s;
    out[b*32+lane]=a>0?(int8_t)fmaxf(-127.f,fminf(127.f,nearbyintf(v/s))):0;
}
/* Bias, activation and residual epilogues avoid extra intermediate tensors. */
__global__ void epilogue(float *x,const float *bias,const float *res,int n,int dim,int activation) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    float v=x[i]+(bias?bias[i%dim]:0.f);
    if (activation) v=gelu(v);
    if (res) v+=res[i];
    x[i]=v;
}
__global__ void swiglu(float *out,const float *gu,int n,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    int row=i/dim,d=i%dim;
    float g=gu[(size_t)row*2*dim+d],u=gu[(size_t)row*2*dim+dim+d];
    out[i]=g/(1.f+expf(-g))*u;
}
/* Both input and output may alias. All reads precede the block barriers. */
template<bool RMS> __global__ void norm(float *out,const float *x,const float *w,
        const float *b,int dim,float eps) {
    int row=blockIdx.x;
    float sum=0, sq=0;
    for(int d=threadIdx.x;d<dim;d+=blockDim.x) {
        float v=x[(size_t)row*dim+d]; sum+=v; sq+=v*v;
    }
    float mean=0;
    if constexpr(!RMS) mean=block_reduce(sum)/dim;
    float var=block_reduce(sq)/dim-mean*mean;
    float inv=rsqrtf(fmaxf(var,0.f)+eps);
    for(int d=threadIdx.x;d<dim;d+=blockDim.x) {
        size_t i=(size_t)row*dim+d;
        out[i]=(x[i]-mean)*inv*w[d]+(b?b[d]:0.f);
    }
}
/* Input is fused Q,K,V; output Q is dense, K/V go directly into the cache. */
__global__ void qk_rope_cache(float *q, float *kc,float *vc,const float *qkv,
        const float *qw,const float *kw,int qheads,int kvheads,int hd,
        const int *position,float eps,float theta) {
    int row=blockIdx.x, h=blockIdx.y, d=threadIdx.x;
    int qdim=qheads*hd, kvdim=kvheads*hd, stride=qdim+2*kvdim;
    bool isq=h<qheads;
    int head=isq?h:h-qheads;
    const float *src=qkv+(size_t)row*stride+(isq?0:qdim)+head*hd;
    __shared__ float vals[128];
    float v=d<hd?src[d]:0;
    float inv=rsqrtf(block_reduce(v*v)/hd+eps);
    if (d<hd) vals[d]=v*inv*(isq?qw[d]:kw[d]);
    __syncthreads();
    if(d>=hd) return;
    int halfdim=hd/2,j=d%halfdim,pos=*position+row;
    float angle=pos*powf(theta,-2.f*j/hd),sn,cs;
    sincosf(angle,&sn,&cs);
    float rot=vals[d]*cs+(d<halfdim?-vals[d+halfdim]:vals[d-halfdim])*sn;
    if(isq) q[(size_t)row*qdim+head*hd+d]=rot;
    else {
        kc[(size_t)pos*kvdim+head*hd+d]=rot;
        vc[(size_t)pos*kvdim+head*hd+d]=qkv[(size_t)row*stride+qdim+kvdim+head*hd+d];
    }
}
/* Bounded-memory online softmax. A wave cooperates on each Q.K dot product;
 * each lane then accumulates a value dimension across the 128-key tile. */
template<int HD,bool CAUSAL> __global__ void attention(float *out,const float *q,
        const float *k,const float *v,int qheads,int kvheads,int seq,int window,
        const int *position) {
    int t=blockIdx.x,h=blockIdx.y,tid=threadIdx.x,lane=tid&31,wave=tid>>5;
    int kvh=h/(qheads/kvheads),ks=kvheads*HD, qs=qheads*HD;
    __shared__ float query[HD],score[128];
    if(tid<HD) query[tid]=q[(size_t)t*qs+h*HD+tid];
    __syncthreads();
    int first,last;
    if constexpr(CAUSAL) { first=0; last=*position+t+1; }
    else { first=t/window*window; last=min(first+window,seq); }
    float m=-INFINITY,l=0,acc=0;
    for(int start=first;start<last;start+=128) {
        for(int j=wave;j<128;j+=4) {
            int key=start+j;
            float dot=0;
            if(key<last) {
                #pragma unroll
                for(int d=lane;d<HD;d+=32)
                    dot=fmaf(query[d],k[(size_t)key*ks+kvh*HD+d],dot);
            }
            dot=wave_sum(dot);
            if(!lane) score[j]=key<last?dot*rsqrtf((float)HD):-INFINITY;
        }
        __syncthreads();
        float next=fmaxf(m,block_reduce<true>(score[tid]));
        float alpha=expf(m-next),p=expf(score[tid]-next);
        float den=block_reduce(p);
        score[tid]=p;
        __syncthreads();
        acc*=alpha;
        if(tid<HD) {
            int count=min(128,last-start);
            for(int j=0;j<count;j++)
                acc=fmaf(score[j],v[(size_t)(start+j)*ks+kvh*HD+tid],acc);
        }
        l=l*alpha+den; m=next;
        __syncthreads();
    }
    if(tid<HD) out[(size_t)t*qs+h*HD+tid]=acc/l;
}
__global__ void split_qkv(float *q,float *k,float *v,const float *src,const float *bias,
        int n,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*dim) return;
    int row=i/dim,d=i%dim; size_t off=(size_t)row*dim*3+d;
    q[i]=src[off]+bias[d]; k[i]=src[off+dim]+bias[dim+d]; v[i]=src[off+2*dim]+bias[2*dim+d];
}
/* im2col produces row-major [output positions, input channels*3*3],
 * suitable for the same FP16 GEMM used by all dense layers. */
__global__ void im2col(half *out,const float *x,int ci,int hi,int wi,int ho,int wo) {
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    int k=ci*9;
    if(idx>=(size_t)ho*wo*k) return;
    int tap=idx%k,p=idx/k,c=tap/9,ky=tap/3%3,kx=tap%3;
    int y=(p/wo)*2+ky-1,xx=(p%wo)*2+kx-1;
    out[idx]=__float2half(y>=0&&y<hi&&xx>=0&&xx<wi?x[((size_t)c*hi+y)*wi+xx]:0.f);
}
/* GEMM output is NHWC; restore CHW for the following convolution. */
__global__ void conv_finish(float *out,const float *x,const float *bias,int channels,int hw) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=channels*hw) return;
    int c=i/hw,p=i%hw;
    out[i]=gelu(x[(size_t)p*channels+c]+bias[c]);
}
__global__ void conv_reshape(float *out,const float *in,int time,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<time*dim) out[i]=in[(i%dim)*time+i/dim];
}
__global__ void position_add(float *x,int n,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*dim) return;
    int p=i/dim,d=i%dim,j=d%(dim/2);
    float angle=p*expf(-j*logf(10000.f)/(dim/2-1));
    x[i]+=d<dim/2?sinf(angle):cosf(angle);
}
__global__ void argmax_stage(float *maxv,int *maxi,const float *x,int n) {
    __shared__ float val[256]; __shared__ int id[256];
    int tid=threadIdx.x,idx=blockIdx.x*256+tid;
    val[tid]=idx<n?x[idx]:-INFINITY; id[tid]=idx;
    __syncthreads();
    for(int d=128;d;d>>=1) {
        if(tid<d && (val[tid+d]>val[tid] || (val[tid+d]==val[tid] && id[tid+d]<id[tid]))) {
            val[tid]=val[tid+d]; id[tid]=id[tid+d];
        }
        __syncthreads();
    }
    if(!tid) { maxv[blockIdx.x]=val[0]; maxi[blockIdx.x]=id[0]; }
}
__global__ void argmax_finish(int *out,const float *values,const int *ids,int n) {
    __shared__ float val[256]; __shared__ int idx[256];
    int tid=threadIdx.x; float best=-INFINITY; int id=0x7fffffff;
    for(int i=tid;i<n;i+=256) if(values[i]>best || (values[i]==best&&ids[i]<id)) {best=values[i];id=ids[i];}
    val[tid]=best;idx[tid]=id;__syncthreads();
    for(int d=128;d;d>>=1) {
        if(tid<d && (val[tid+d]>val[tid] || (val[tid+d]==val[tid]&&idx[tid+d]<idx[tid]))) {val[tid]=val[tid+d];idx[tid]=idx[tid+d];}
        __syncthreads();
    }
    if(!tid)*out=idx[0];
}
/* 128-byte coalesced wave transactions, four independent FMA chains. */
template<bool Q8> __global__ void gemv4(float *out,const half *w,const int8_t *w8,
        const float *scales,const float *x,int in,int rows) {
    extern __shared__ float sx[];
    int tid=threadIdx.x,lane=tid&31,row=blockIdx.x*8+(tid>>5);
    for(int i=tid;i<in;i+=256)sx[i]=x[i];
    __syncthreads();if(row>=rows)return;
    float a=0,b=0,c=0,d=0;
    for(int i=lane*4;i<in;i+=128) {
        size_t off=(size_t)row*in+i;
        float4 xv=*reinterpret_cast<const float4*>(sx+i);
        float4 v;
        if constexpr(Q8) {
            int packed=*reinterpret_cast<const int*>(w8+off);float scale=scales[off/32];
            v=make_float4((int8_t)packed*scale,(int8_t)(packed>>8)*scale,(int8_t)(packed>>16)*scale,(int8_t)(packed>>24)*scale);
        }else {
            uint2 packed=*reinterpret_cast<const uint2*>(w+off);
            v=make_float4(__half2float(__ushort_as_half(packed.x&65535)),__half2float(__ushort_as_half(packed.x>>16)),__half2float(__ushort_as_half(packed.y&65535)),__half2float(__ushort_as_half(packed.y>>16)));
        }
        a=fmaf(v.x,xv.x,a);b=fmaf(v.y,xv.y,b);c=fmaf(v.z,xv.z,c);d=fmaf(v.w,xv.w,d);
    }
    float sum=wave_sum((a+b)+(c+d));if(!lane)out[row]=sum;
}
