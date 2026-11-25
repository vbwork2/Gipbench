#!/usr/bin/env python3
"""Hợp đồng adapter (infer.adapters.base + bench.transcribe), dùng adapter giả: không cần model / GPU / mạng.

  python tests/infer/test_adapters.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from gipbench.bench.transcribe import guard, subsample
from gipbench.infer.adapters import build_adapter
from gipbench.infer.adapters.base import ASRAdapter, AdapterInfo, run_manifest, split_long
from gipbench.infer.adapters.sherpa_transducer import SherpaTransducerAdapter

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


class Fake(ASRAdapter):
    def __init__(self, fail_at=None, wrong_count=False):
        self.info = AdapterInfo(system="fake", engine="fake", model_id="fake", revision="0", inference_mode="open_vocab")
        self.calls, self.loaded, self.fail_at, self.wrong = [], 0, fail_at, wrong_count

    def load(self):
        self.loaded += 1

    def transcribe_batch(self, paths):
        self.calls.append(list(paths))
        if self.fail_at and any(self.fail_at in p for p in paths):
            raise RuntimeError("lỗi giả")
        out = [Path(p).stem.upper() for p in paths]
        return out[:-1] if self.wrong else out


def rows(n):
    return [{"id": f"u{i}", "_audio_path": f"/x/u{i}.wav", "duration": 2.0, "dataset": "a" if i % 2 else "b"}
            for i in range(n)]


def test_streaming_tail(work_dir):
    class Recognizer:
        def __init__(self):
            self.calls = []
            self.finished = self.decoded = False

        def create_stream(self):
            return self

        def accept_waveform(self, rate, samples):
            self.calls.append((rate, len(samples), bool(np.any(samples))))

        def input_finished(self):
            self.finished = True

        def is_ready(self, stream):
            return stream.finished and not stream.decoded

        def decode_stream(self, stream):
            stream.decoded = True

        def get_result(self, stream):
            return "flushed" if stream.decoded else ""

    adapter = SherpaTransducerAdapter({
        "system": "test", "model": {"id": "test", "dir": str(work_dir)}, "inference_mode": "open_vocab",
        "decode": {"streaming": True, "feed_ms": 100, "tail_pad_ms": 660},
    }, work_dir)
    adapter.rec = Recognizer()
    result = adapter.transcribe_stream(np.ones(1900, dtype=np.float32), 16000)
    check("Streaming adds the configured silence before draining the final frames",
          result == "flushed" and adapter.rec.calls == [(16000, 1600, True), (16000, 300, True), (16000, 10560, False)])


def main():
    d = Path(tempfile.mkdtemp())
    test_streaming_tail(d)
    meta = run_manifest(Fake(), rows(7), d / "r1", batch=3)
    hy = [json.loads(l) for l in open(d / "r1" / "hyps.jsonl", encoding="utf-8")]
    check("một hyp mỗi id, đúng thứ tự manifest", [h["id"] for h in hy] == [f"u{i}" for i in range(7)])
    check("run_meta có rtf / hardware / sha256", meta["n_utts"] == 7 and "hardware" in meta and meta["hyps_sha256"]
          and json.loads((d / "r1" / "run_meta.json").read_text(encoding="utf-8"))["inference_mode"] == "open_vocab")
    # resume: lần 1 lỗi ở u5 -> giữ hyp đã xong; lần 2 chỉ giải mã phần còn thiếu
    try:
        run_manifest(Fake(fail_at="u5"), rows(7), d / "r2", batch=2)
        check("lỗi adapter làm dừng (không ghi hyp rỗng)", False)
    except RuntimeError:
        check("lỗi adapter làm dừng (không ghi hyp rỗng)", not (d / "r2" / "hyps.jsonl").exists())
    f2 = Fake()
    m2 = run_manifest(f2, rows(7), d / "r2", batch=2)
    check("resume chỉ giải mã câu còn thiếu", m2["n_resumed"] == 4 and m2["n_decoded_this_run"] == 3
          and sum(len(c) for c in f2.calls) == 3, m2)
    f3 = Fake()
    run_manifest(f3, rows(7), d / "r2", batch=2)
    check("đã đủ hyp thì không nạp model", f3.loaded == 0 and not f3.calls)
    try:
        run_manifest(Fake(wrong_count=True), rows(3), d / "r3", batch=3)
        check("adapter trả thiếu hyp -> lỗi", False)
    except RuntimeError:
        check("adapter trả thiếu hyp -> lỗi", True)
    try:
        run_manifest(Fake(), rows(2) + rows(1), d / "r4")
        check("manifest trùng id -> lỗi", False)
    except ValueError:
        check("manifest trùng id -> lỗi", True)
    # luật inference mode
    for kw, name in [(dict(inference_mode="open_vocab", context_vocab={"provenance": "external"}), "open_vocab + hotwords bị cấm"),
                     (dict(inference_mode="context_bias", context_vocab={"provenance": "test_derived"}),
                      "hotwords test_derived cần oracle_bias"),
                     (dict(inference_mode="context_bias"), "context_bias cần context_vocab"),
                     (dict(inference_mode="zero_shot"), "mode lạ bị cấm")]:
        try:
            AdapterInfo(system="s", engine="e", model_id="m", revision="r", **kw)
            check(name, False)
        except ValueError:
            check(name, True)
    AdapterInfo(system="s", engine="e", model_id="m", revision="r", inference_mode="oracle_bias",
                context_vocab={"provenance": "test_derived"})
    check("oracle_bias + test_derived hợp lệ", True)
    # cắt audio dài
    x = np.zeros(16000 * 65, dtype=np.float32)
    check("cắt 65 s thành 30+30+5 s", [len(p) // 16000 for p in split_long(x, 16000, 30)] == [30, 30, 5])
    check("không cắt khi max_sec=None", len(split_long(x, 16000, None)) == 1)
    # bench.transcribe: chặn oracle trên G0, chặn hotwords từ catalog_from_bench khai báo sai
    g0 = {"_name": "g0", "allowed_inference_modes": ["open_vocab", "lm"]}
    check("G0 chặn oracle_bias", guard({"inference_mode": "oracle_bias"}, g0) != [])
    cat = d / "cat.tsv"
    cat.write_text("term\n", encoding="utf-8")
    v0 = {"_name": "v0", "allowed_inference_modes": ["open_vocab", "context_bias", "oracle_bias"], "catalog": str(cat),
          "catalog_from_bench": True}
    check("catalog_from_bench + provenance external bị chặn",
          guard({"inference_mode": "context_bias", "context_vocab": {"catalog": str(cat), "provenance": "external"}}, v0) != [])
    check("catalog_from_bench + oracle_bias/test_derived được chạy",
          guard({"inference_mode": "oracle_bias", "context_vocab": {"catalog": str(cat), "provenance": "test_derived"}}, v0) == [])
    sub = subsample(rows(20), 3)
    check("subsample: N câu mỗi bộ, cố định", len(sub) == 6 and sub == subsample(rows(20), 3))
    try:
        build_adapter({"engine": "khong_co"}, d)
        check("engine lạ -> lỗi", False)
    except ValueError:
        check("engine lạ -> lỗi", True)
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
