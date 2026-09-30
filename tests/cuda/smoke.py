#!/usr/bin/env python3
"""End-to-end CUDA checks: CLI, stdin, language control, graphs and streaming."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import wave


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="./qwen_asr_cuda")
    p.add_argument("--model", default="qwen3-asr-0.6b")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sample = root / "samples/jfk.wav"
    def run(options, data=None, env=None, success=True):
        cp = subprocess.run([args.binary, "-d", args.model, *options], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ, "QWEN_CUDA_Q8": "1",
                                 "QWEN_CUDA_GRAPHS": "1", **(env or {})}, timeout=120)
        if success and cp.returncode:
            raise AssertionError(cp.stderr.decode(errors="replace"))
        if not success and cp.returncode == 0:
            raise AssertionError("Invalid input accepted")
        return cp.stdout.decode().strip(), cp.stderr.decode()
    text, _ = run(["--list-languages"])
    assert len(text.split(",")) == 30 and "Italian" in text, text
    print("PASS supported language list", flush=True)
    baseline, stderr = run(["-i", str(sample), "--silent", "--precision", "q8"])
    assert "ask" in baseline.lower() and "country" in baseline.lower(), baseline
    assert not stderr, stderr
    print("PASS file transcription and --silent", flush=True)
    for description, data in [("WAV stdin", sample.read_bytes())]:
        text, _ = run(["--stdin", "--silent"], data)
        assert text == baseline
        print(f"PASS {description}", flush=True)
    with wave.open(str(sample), "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        pcm = wav.readframes(wav.getnframes())
    text, _ = run(["--stdin", "--silent"], pcm)
    assert text == baseline
    print("PASS raw PCM stdin", flush=True)
    text, _ = run(["-i", str(sample), "--silent"], env={"QWEN_CUDA_GRAPHS": "0"})
    assert text == baseline
    print("PASS graph replay equals direct launches", flush=True)
    text, stderr = run(["-i", str(sample), "--stream", "--debug"])
    assert "country" in text.lower(), text
    assert stderr.count("  Commit:") >= 2, stderr
    print("PASS interactive streaming and prefix reuse", flush=True)
    text, _ = run(["-i", str(sample), "--silent", "--precision", "fp16"])
    assert "country" in text.lower(), text
    print("PASS FP16 decoder", flush=True)
    text, _ = run(["-i", str(sample), "--silent", "--language", "English"])
    assert "country" in text.lower(), text
    print("PASS forced language", flush=True)
    for options in [["-i", str(sample), "--precision", "bad"], ["--precision"], ["-i", str(sample), "--stdin"], ["-i", "/nonexistent/qwen-asr.wav"]]:
        run(options, success=False)
    print("PASS invalid CLI and missing audio", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, n in [("silence", 16000), ("empty", 0)]:
            path = Path(tmp) / (name + ".wav")
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * n)
            text, _ = run(["-i", str(path), "--silent"], success=(n > 0))
            assert text == "", (name, text)
    print("PASS silence and rejection of empty WAV", flush=True)
    long_sample = root / "samples/night_of_the_living_dead_1968/45s_dont_be_afraid_of_me.wav"
    text, _ = run(["-i", str(long_sample), "--silent", "-S", "20"])
    assert "helen" in text.lower() and "cooper" in text.lower(), text
    print("PASS 45-second segmented audio", flush=True)

if __name__ == "__main__":
    main()
