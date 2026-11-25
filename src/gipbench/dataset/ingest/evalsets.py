#!/usr/bin/env python3
"""Nhập các bộ test ASR tiếng Việt công khai bổ sung (track E0: so với số đã công bố, vd bảng của model card VieGem).

  python -m gipbench.dataset.ingest.evalsets --sets fleurs_vi cv17_vi vietmed lsvsc vimd speech_massive_vi vimedcss_test \
      [--out <data_root>/dl/evalsets] [--keep-source] [--limit N]

Mỗi bộ: tải đúng các file nguồn ở revision cố định (huggingface_hub, local_dir), kiểm sha256 với bản khoá trong SPECS,
tách audio + lời thoại GỐC (không sửa chữ), ghi:
  <out>/<bộ>/audio/<id>.<ext>   audio: giữ nguyên byte nếu đã là mono <= 16 kHz hoặc định dạng nén; còn lại (44.1 / 48 kHz,
                                stereo) resample soxr HQ 16 kHz mono PCM16 (`audio_derivation: soxr_hq_16k_mono_pcm16`)
  <out>/<bộ>/rows.jsonl          id, dataset, upstream_id, upstream_split, text_tagged (= lời thoại gốc), speaker, ...,
                                audio (tương đối so với rows.jsonl), audio_sha256, duration (đo từ audio)
  <out>/<bộ>/ingest_meta.json    file nguồn + sha256, số dòng, số giờ, dòng bị bỏ kèm lý do
File nguồn bị xoá sau khi tách (đỡ tốn ổ), trừ khi --keep-source. Trước mỗi lần tải kiểm dung lượng trống.
Dùng lại trong gipbench.dataset.release.fetch (kind "evalset") để người nhận bản share dựng lại đúng audio.
"""
import argparse
import csv
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from gipbench.core.fileio import sha256_file
from gipbench.dataset.ingest.common import safe

DERIVED_TAG = "soxr_hq_16k_mono_pcm16"

# Bản khoá nguồn. sha256 của file LFS lấy từ API Hugging Face; file không LFS (tsv) tính khi tải lần đầu (17/11/2026).
SPECS: Dict[str, Dict] = {
    "fleurs_vi": {
        "repo": "google/fleurs", "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd", "license": "CC-BY-4.0",
        "split": "test", "reader": "fleurs", "expected": 857,
        "files": [("data/vi_vn/test.tsv", "eb744c2be5f677b49c35527d428a4f647294bd0f67abd790c296e91efaaffb93"),
                  ("data/vi_vn/audio/test.tar.gz", "b359261216ef14b8eea159abce794e270c2a0be903218cf1e1e8f7c7c1565b5c")]},
    "cv17_vi": {
        # bản mirror cộng đồng của Common Voice 17.0 (Mozilla đã gỡ bản chính thức khỏi HF); CC0
        "repo": "fsicoli/common_voice_17_0", "revision": "8262c16bf297c87a9cd88c51997c4758ed7a8ba2", "license": "CC0-1.0",
        "split": "test", "reader": "cv17", "expected": 1274,
        "files": [("transcript/vi/test.tsv", "c2bf109b52e6348a6bc9b1410877feed20e029094516599c31e80e62df581fa5"),
                  ("audio/vi/test/vi_test_0.tar", "a4460f5c1efdf106c7c19cf30444958582532b2def431598aa90c78207625518")]},
    "vietmed": {
        "repo": "leduckhai/VietMed", "revision": "f302146de79c5cbd6a7d14400fefa2c8cfda7738", "license": "MIT (theo HF)",
        "split": "test", "reader": "parquet", "expected": 3437,
        "files": [("data/test-00000-of-00001.parquet", "275b205059e1e4e0ef88eb7bae3cf14287de456752e74a57e0c10476725b4a22")],
        "id": "utterance_id", "text": "text", "speaker": "speaker_name", "session": "audio_name",
        "meta": {"gender": "gender", "accent": "accent", "role": "role", "rec_condition": "rec_condition"}, "domain": "medical"},
    "lsvsc": {
        "repo": "doof-ferb/LSVSC", "revision": "269de0604fd94765648cbfcaed4341ef65a31e50", "license": "CC-BY-4.0",
        "split": "test", "reader": "parquet", "expected": 5683,
        "files": [("data/test-00000-of-00003.parquet", "24a086a90b5c5f7072e1652b15215164fea9de017f60f9ab58836406116b9b30"),
                  ("data/test-00001-of-00003.parquet", "51bc7fc3483aa958f9cf4773d9da1b219f694c93f977ac29b99ecceb5ef74657"),
                  ("data/test-00002-of-00003.parquet", "1444dcac1ea83c8ecf9dfea461135eab1d582065967c7a14936363bf88a1df53")],
        "id": None, "id_with_shard": True, "text": "transcription", "speaker": None,
        "meta": {"gender": "gender", "dialect": "dialect", "topic": "topic", "emotion": "emotion", "age": "age"}},
    "vimd": {
        "repo": "nguyendv02/ViMD_Dataset", "revision": "3a5b30157034e7eadd5c75fae1a820c6f9383398",
        "license": "CC-BY-NC-ND-4.0", "split": "test", "reader": "parquet", "expected": 2026,  # tổng num_rows 14 shard
        "files": [(f"data/test-{i:05d}-of-00014.parquet", s) for i, s in enumerate([
            "e3df51b347bb4909b01083d7e134206553deae1afb4f1430ff69340ea69d6369",
            "0520fb3d6666e1804ad5d65529bc21c93628b3cc928bcc82276d72b8b8458d6c",
            "c484bb6496563d5d62fe111986d6bb71446bf2de80a8aad2e3afe4aec56fd1db",
            "a7e24f9f4fc7b9be63409103eb41451a4199c49ca51f7cf12ef448f99fc13518",
            "560557ff21b7d1ee0966e7040750b6fab1b59e4e21acce727c732d3e5802ff57",
            "e9890171776d437608a8612b670ef263c0cbc4cce32661364589e48f3c90228c",
            "965f53016e364c732090db2aa9510f12ac26d89c5236dcf5ea8cd2cce0601a58",
            "1b2090a1b93c657e9c5a1a564ab289eefc3fdcdaaf7ee000bdd4117b2175f84f",
            "d617472eb2880ed1c74352381f3158cf77891694e6647f4c01dd547ce18107eb",
            "fadb8b19c8bc2bc62165de5bbf4066b72351a5b9b025d129c725c7f8adff4f8d",
            "32ae6182cdc09e91f65f2b5867ebbc0fe19ceb2016ad59f57e5fc626319a8fb4",
            "2cd7184e665811b6df5ecabad4c72ca21d8f89ed32aa7a59e590cd95ed44cbc5",
            "ee1cfccae0618ec5cb6e9053e2e9ee4dcb45a163d2755e281720f981c4c97826",
            "4f83997a860facdfab0bd3f12d0ce9a67afdac309bc7cb77359eb4d92771fdab"])],
        "id": "filename", "text": "text", "speaker": "speakerID",
        "meta": {"gender": "gender", "region": "region", "province": "province_name"}},
    "speech_massive_vi": {
        # split validation (dev) của Speech-MASSIVE vi-VN; split test nằm ở repo gated FBK-MT/Speech-MASSIVE-test
        "repo": "FBK-MT/Speech-MASSIVE", "revision": "ff792febc16187a21e5bca38fb02a55daf91dc05", "license": "CC-BY-NC-SA-4.0",
        "split": "validation", "reader": "parquet", "expected": 2033,
        "files": [("vi-VN/validation-00000-of-00002.parquet", "bcd5f497b49baf9dd5672e082c4e8d431dae410c19009615bbeeced616eeaf20"),
                  ("vi-VN/validation-00001-of-00002.parquet", "0392e589f7de814ce71755743ac8413d3d7aea5a9f6b605f884c4d344702586b")],
        "id": "path", "text": "utt", "speaker": "speaker_id",
        "meta": {"gender": "speaker_sex", "age": "speaker_age", "scenario": "scenario_str", "massive_id": "id",
                 "is_validated": "is_validated", "residence": "speaker_country_of_residence"}, "domain": "voice_assistant"},
    "vimedcss_test": {
        "repo": "tensorxt/ViMedCSS", "revision": "b6959a18a08739464733930a872e7125c03e6558", "license": "CC-BY-4.0",
        "split": "test", "reader": "parquet", "expected": 1614,
        "files": [(f"data/test-{i:05d}-of-00005.parquet", s) for i, s in enumerate([
            "be4f3650414cdf7181a1348789349d555ba8cf25a2e588ff59bc24af49550369",
            "c8774838bb5b28812e1e0f6dd6f20256d5073a6cc7b82a6277eeac8e31e58b72",
            "bcd91d2bf99fad7cbbe8d8708c43c1f5206d2606f8f6963293dc6e403b45b07b",
            "8cc20f58721a07682dba7abc05bcee9fd4742ecc6cc0f6b550dd4edfd0cd3fa7",
            "6f073c4792ae4dc2fc94021be51e5ab5592499a1a241754bdbedb88d8f808062"])],
        "id": "segment_id", "text": "segment_text", "speaker": "original_video_link",
        "meta": {"topic": "topic", "cs_terms_list": "cs_terms_list"}, "domain": "medical"},
    # --- bộ gated (cần token HF đã đồng ý điều khoản trên trang dataset) ---
    "phoaudiobook": {
        # điều khoản: chỉ nghiên cứu / giáo dục, KHÔNG phân phối lại bất kỳ phần nào -> bản share không chứa lời thoại
        "repo": "thivux/phoaudiobook", "revision": "f19f2410a0497c32f3dc8d9a054d1a14a255b9ee",
        "license": "PhoAudiobook terms (gated): chỉ nghiên cứu / giáo dục, không phân phối lại, trích dẫn ACL 2025",
        "split": "test", "reader": "parquet", "expected": 383, "redistribute_text": False,
        "files": [("data/test-00000-of-00001.parquet", "b28749011ca53bb46cf7bfa50f272773feccbfb57fcf9313ea209f65a34b56f6")],
        "id": None, "id_with_shard": True, "text": "text", "speaker": "speaker", "domain": "audiobook"},
    "bud500": {
        "repo": "linhtran92/viet_bud500", "revision": "4b30571f395781fddc3a4946fb378648f8572714", "license": "CC-BY-NC-SA-4.0",
        "split": "test", "reader": "parquet", "expected": 7500,
        "files": [("data/test-00000-of-00002-531c1d81edb57297.parquet", "a204d93628d280141aea359cb37c94eef8eec7268722e6c2e22d373f47f3bcfe"),
                  ("data/test-00001-of-00002-1b2cd3ea16a2c5b1.parquet", "e5eb289a84ef548dd53f8117f4afdc9c74a2e5a78a1d16b9957ba926d7eec56b")],
        "id": None, "id_with_shard": True, "text": "transcription", "speaker": None},
    "gigaspeech2_vi": {
        # audio YouTube, SpeechColab không sở hữu bản quyền; chỉ nghiên cứu phi thương mại -> bản share không chứa lời thoại
        "repo": "speechcolab/gigaspeech2", "revision": "8dc0d0e502b7e6d5649a13cd4b677cf10c0b21e3",
        "license": "GigaSpeech2 terms (gated; metadata Apache-2.0, audio YouTube): nghiên cứu / giáo dục phi thương mại",
        "split": "test", "reader": "tsv_tar", "expected": 4422, "redistribute_text": False,
        "files": [("data/vi/test.tsv", "287de4473efe384ebecd123d8d1fae7d3a8550bc8e06af4a5879a61ed540aa95"),
                  ("data/vi/test.tar.gz", "c1963f95ef8351366ec942c20328300cfb06a3ce750c0ae42db3f67d38f0354b")],
        "domain": "youtube"},
    "speech_massive_test_vi": {
        "repo": "FBK-MT/Speech-MASSIVE-test", "revision": "2e6d34fb35e3ac7021980fb018e7894b151a5c8e",
        "license": "CC-BY-NC-SA-4.0", "split": "test", "reader": "parquet", "expected": 2974,
        "files": [("vi-VN/test-00000-of-00003.parquet", "6063850c3f138ddb93759a8b3cfe9973f4ac441511aced0aa908c44585f2abc5"),
                  ("vi-VN/test-00001-of-00003.parquet", "2c73016b2aa3ce6f3e1fbea228624645a31e37321e21e4bf1e4a69adef09a1a7"),
                  ("vi-VN/test-00002-of-00003.parquet", "d0bea81e8a787e92b8e0dafae2dfd86067635f3e620d51ad85e3617c37d1a12b")],
        "id": "path", "text": "utt", "speaker": "speaker_id",
        "meta": {"gender": "speaker_sex", "age": "speaker_age", "scenario": "scenario_str", "massive_id": "id",
                 "is_validated": "is_validated", "residence": "speaker_country_of_residence"}, "domain": "voice_assistant"},
}
# Trường lời thoại bị bỏ khỏi bản share khi bộ không cho phân phối lại (fetch lấy lại từ nguồn, cùng giá trị)
TEXT_FIELDS = ("text_tagged", "text_orig", "text_norm_upstream")


# ------------------------------------------------------------------ đọc nguồn -> (upstream_id, meta, bytes, ext)
def _str(v):
    return None if v is None else str(v)


def read_parquet(spec: Dict, files: List[str], limit: int = 0) -> Iterator[Tuple[str, Dict, bytes, str]]:
    import pyarrow.parquet as pq
    n = 0
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for ri, r in enumerate(pf.read_row_group(rg).to_pylist()):
                a = r.get("audio") or {}
                path = a.get("path") or ""
                up = _str(r.get(spec["id"])) if spec.get("id") else path
                if not up:
                    up = f"{Path(f).stem}:{rg}:{ri}"
                elif spec.get("id_with_shard"):  # tên file audio chỉ duy nhất trong từng shard
                    up = f"{Path(f).name.split('-of-')[0]}/{up}"
                meta = {"text": (r.get(spec["text"]) or "").strip(),
                        "speaker": _str(r.get(spec["speaker"])) if spec.get("speaker") else None,
                        "session": _str(r.get(spec["session"])) if spec.get("session") else None}
                meta.update({k: _str(r.get(c)) for k, c in (spec.get("meta") or {}).items()})
                yield up, meta, a.get("bytes") or b"", Path(path).suffix or ".wav"
                n += 1
                if limit and n >= limit:
                    return


def read_fleurs(spec: Dict, files: List[str], limit: int = 0):
    tsv, tgz = files
    rows = {}
    with open(tsv, encoding="utf-8", newline="") as f:
        for c in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id (câu FLEURS), file_name, raw_transcription, transcription, characters, num_samples, gender
            rows[c[1]] = {"text": c[2].strip(), "text_norm_upstream": c[3].strip(), "sentence_id": c[0], "gender": c[6]}
    n = 0
    archive = (tarfile.open(fileobj=tgz, mode="r|gz") if hasattr(tgz, "read") else tarfile.open(tgz, "r:gz"))
    with archive as tf:
        for m in tf:
            name = Path(m.name).name
            if m.isfile() and name in rows:
                with tf.extractfile(m) as fh:
                    yield name, dict(rows[name], speaker=None, session=rows[name]["sentence_id"]), fh.read(), ".wav"
                n += 1
                if limit and n >= limit:
                    return


def read_cv17(spec: Dict, files: List[str], limit: int = 0):
    tsv, tar = files
    rows = {}
    with open(tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows[r["path"]] = {"text": (r["sentence"] or "").strip(), "speaker": f"cv_{r['client_id'][:16]}",
                               "session": None, "gender": r.get("gender") or None, "age": r.get("age") or None,
                               "accent": r.get("accents") or None, "sentence_id": r.get("sentence_id")}
    n = 0
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            name = Path(m.name).name
            if m.isfile() and name in rows:
                with tf.extractfile(m) as fh:
                    yield name, rows[name], fh.read(), ".mp3"
                n += 1
                if limit and n >= limit:
                    return


def read_tsv_tar(spec: Dict, files: List[str], limit: int = 0):
    """GigaSpeech2: tsv không header `<id>\\t<text>`; tar.gz có file audio tên <id>.<ext> (tìm theo stem)."""
    tsv, tgz = files
    rows = {}
    with open(tsv, encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                k, _, t = line.partition("\t")
                rows[k] = {"text": t.strip(), "speaker": None, "session": None}
    n = 0
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = Path(m.name).stem, Path(m.name).suffix
            if stem in rows:
                with tf.extractfile(m) as fh:
                    yield stem, rows[stem], fh.read(), ext or ".wav"
                n += 1
                if limit and n >= limit:
                    return


READERS = {"parquet": read_parquet, "fleurs": read_fleurs, "cv17": read_cv17, "tsv_tar": read_tsv_tar}


# ------------------------------------------------------------------ ghi audio
def write_audio(data: bytes, ext: str, out_stem: Path) -> Tuple[Path, str, float]:
    """Giữ nguyên byte nếu là định dạng nén hoặc WAV mono <= 16 kHz; còn lại resample 16 kHz mono PCM16."""
    import soundfile as sf
    info = sf.info(io.BytesIO(data))
    keep = ext.lower() in (".mp3", ".ogg", ".opus", ".flac") or (info.samplerate <= 16000 and info.channels == 1)
    if keep:
        p = out_stem.with_suffix(ext.lower())
        p.write_bytes(data)
        return p, "upstream_bytes", round(info.frames / info.samplerate, 3)
    import numpy as np
    import soxr
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = soxr.resample(wav, sr, 16000, quality="HQ")
    p = out_stem.with_suffix(".wav")
    sf.write(str(p), np.asarray(wav), 16000, subtype="PCM_16")
    return p, DERIVED_TAG, round(len(wav) / 16000, 3)


def materialize(key: str, files: List[str], audio_dir: Path, id_prefix: str = "e0_",
                want: Optional[set] = None, limit: int = 0) -> Iterator[Tuple[Dict, Optional[str]]]:
    """Tách audio. Trả về (dòng, lý do bỏ hoặc None). want = tập upstream_id cần (fetch); None = mọi dòng."""
    spec = SPECS[key]
    audio_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for up, meta, data, ext in READERS[spec["reader"]](spec, files, limit):
        if want is not None and up not in want:
            continue
        rid = f"{id_prefix}{key}_{safe(up)}"
        if rid in seen:
            yield {"id": rid, "upstream_id": up}, "id trùng"
            continue
        seen.add(rid)
        if not data:
            yield {"id": rid, "upstream_id": up}, "không có audio"
            continue
        p, deriv, dur = write_audio(data, ext, audio_dir / rid)
        row = base_row(key, rid, up, meta)
        row.update({"audio_derivation": deriv, "duration": dur, "_path": str(p), "audio_sha256": sha256_file(p)})
        yield row, (None if row["text_tagged"] else "text rỗng")


def base_row(key: str, rid: str, up: str, meta: Dict) -> Dict:
    """Phần dòng manifest KHÔNG phụ thuộc audio (dùng chung cho nhập đầy đủ và --metadata-only)."""
    spec = SPECS[key]
    meta = dict(meta)
    row = {"id": rid, "dataset": key, "upstream_id": up, "upstream_split": spec["split"],
           "text_tagged": meta.pop("text"), "speaker": meta.pop("speaker") or rid,
           "source": meta.pop("session") or meta.get("sentence_id") or rid,
           "domain": spec.get("domain") or "general", "license": spec["license"],
           "annotation_status": "upstream_human"}
    if spec.get("redistribute_text") is False:
        row["text_redistributable"] = False  # package bỏ lời thoại; fetch lấy lại từ nguồn
    row.update({k: v for k, v in meta.items() if v not in (None, "")})
    return row


METADATA_DERIVATION = "pinned_source_extract"  # audio tách từ file nguồn khoá sha256; không ghi sha256 từng file


def read_metadata(key: str, id_prefix: str = "e0_") -> Iterator[Tuple[Dict, Optional[str]]]:
    """--metadata-only: dòng manifest (id, lời thoại, meta) đọc từ nguồn ở revision khoá MÀ KHÔNG tải audio.
    Parquet: đọc mọi cột trừ audio (+ audio.path) qua HfFileSystem; tsv + tar (GigaSpeech2): đọc tsv.
    Không có audio_sha256 / duration đo được: fetch kiểm toàn vẹn bằng sha256 file nguồn."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download
    spec = SPECS[key]
    seen = set()

    def emit(up, meta):
        rid = f"{id_prefix}{key}_{safe(up)}"
        if rid in seen:
            return {"id": rid, "upstream_id": up}, "id trùng"
        seen.add(rid)
        row = base_row(key, rid, up, meta)
        row["audio_derivation"] = METADATA_DERIVATION
        return row, (None if row["text_tagged"] else "text rỗng")

    if spec["reader"] == "parquet":
        fs = HfFileSystem()
        for fn, _ in spec["files"]:
            with fs.open(f"datasets/{spec['repo']}@{spec['revision']}/{fn}", "rb", block_size=256 << 10,
                         cache_type="readahead") as f:
                pf = pq.ParquetFile(f)
                cols = [c for c in pf.schema_arrow.names if c != "audio"]
                for rg in range(pf.num_row_groups):
                    t = pf.read_row_group(rg, columns=cols + ["audio.path"]).to_pylist()
                    for ri, r in enumerate(t):
                        path = (r.get("audio") or {}).get("path") or ""
                        up = _str(r.get(spec["id"])) if spec.get("id") else path
                        if not up:
                            up = f"{Path(fn).stem}:{rg}:{ri}"
                        elif spec.get("id_with_shard"):
                            up = f"{Path(fn).name.split('-of-')[0]}/{up}"
                        meta = {"text": (r.get(spec["text"]) or "").strip(),
                                "speaker": _str(r.get(spec["speaker"])) if spec.get("speaker") else None,
                                "session": _str(r.get(spec["session"])) if spec.get("session") else None}
                        meta.update({k: _str(r.get(c)) for k, c in (spec.get("meta") or {}).items()})
                        yield emit(up, meta)
    elif spec["reader"] == "tsv_tar":
        tsv = hf_hub_download(spec["repo"], spec["files"][0][0], repo_type="dataset", revision=spec["revision"])
        if sha256_file(tsv) != spec["files"][0][1]:
            raise SystemExit(f"{key}: tsv khác bản khoá")
        for line in open(tsv, encoding="utf-8"):
            line = line.rstrip("\n")
            if line:
                k, _, t = line.partition("\t")
                yield emit(k, {"text": t.strip(), "speaker": None, "session": None})
    else:
        raise SystemExit(f"{key}: --metadata-only chưa hỗ trợ reader {spec['reader']} (bộ nhỏ: nhập đầy đủ)")


def download(key: str, src_dir: Path, min_free_gb: float = 3.0, only: Optional[List] = None) -> List[str]:
    from huggingface_hub import hf_hub_download
    spec = SPECS[key]
    out = []
    src_dir.mkdir(parents=True, exist_ok=True)
    for fn, want in (only or spec["files"]):
        free = shutil.disk_usage(src_dir).free / 1e9
        if free < min_free_gb:
            raise SystemExit(f"còn {free:.1f} GB trống (< {min_free_gb} GB): dừng trước khi tải {fn}")
        p = hf_hub_download(spec["repo"], fn, repo_type="dataset", revision=spec["revision"], local_dir=str(src_dir))
        got = sha256_file(p)
        if got != want:
            raise SystemExit(f"{spec['repo']}/{fn}: sha256 {got[:12]} khác bản khoá {want[:12]}")
        out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", help="mặc định <data_root>/dl/evalsets")
    ap.add_argument("--keep-source", action="store_true", help="không xoá file nguồn sau khi tách")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N dòng đầu mỗi bộ (thử nhanh)")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--metadata-only", action="store_true",
                    help="chỉ đọc id + lời thoại + meta từ nguồn (không tải audio) -> <bộ>/rows_meta.jsonl")
    args = ap.parse_args(argv)
    from gipbench.core import paths as P
    root = Path(args.out or Path(P.path("data_root")) / "dl" / "evalsets")
    if args.metadata_only:
        for key in args.sets:
            d = root / key
            d.mkdir(parents=True, exist_ok=True)
            rows, dropped = [], []
            for row, why in read_metadata(key):
                (dropped.append({"id": row["id"], "upstream_id": row["upstream_id"], "reason": why}) if why
                 else rows.append(row))
            with open(d / "rows_meta.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            exp = SPECS[key].get("expected")
            (d / "ingest_meta_metadata.json").write_text(json.dumps(
                {"set": key, "mode": "metadata_only", "repo": SPECS[key]["repo"], "revision": SPECS[key]["revision"],
                 "files": [{"file": fn, "sha256": s} for fn, s in SPECS[key]["files"]], "rows": len(rows),
                 "expected": exp, "dropped": dropped}, ensure_ascii=False, indent=1), encoding="utf-8")
            flag = "" if exp in (None, len(rows)) else f"  <-- KHÁC số công bố {exp}"
            print(f"{key} (metadata): {len(rows)} câu, bỏ {len(dropped)}{flag}", flush=True)
        return
    for key in args.sets:
        spec = SPECS[key]
        d = root / key
        src = root / "_src" / key
        t0 = time.time()
        rows, dropped, seen = [], [], set()
        # parquet: từng shard một (tải -> tách -> xoá) để không cần chỗ cho cả bộ; tsv + tar: tải cùng lúc
        batches = [[f] for f in spec["files"]] if spec["reader"] == "parquet" else [spec["files"]]
        for bi, b in enumerate(batches):
            if args.limit and len(rows) >= args.limit:
                break
            # chạy tiếp được: shard đã tách xong (cùng sha256 nguồn) được nạp lại, không tải lại
            part = d / f".part{bi:03d}.{b[0][1][:12]}.jsonl"
            if part.exists() and not args.limit:
                done = json.loads(part.read_text(encoding="utf-8"))
                for row in done["rows"]:
                    if row["id"] in seen:
                        raise SystemExit(f"{key}: id {row['id']} trùng giữa các shard, cần upstream_id có tên shard")
                    seen.add(row["id"])
                rows += done["rows"]
                dropped += done["dropped"]
                print(f"  {key}: shard {bi + 1}/{len(batches)} đã có ({len(done['rows'])} câu), bỏ qua tải", flush=True)
                continue
            files = download(key, src, args.min_free_gb, only=b)
            brows, bdrop = [], []
            for row, why in materialize(key, files, d / "audio", limit=args.limit):
                if not why and row["id"] in seen:  # trùng giữa hai shard: file audio vừa ghi đè bản trước
                    raise SystemExit(f"{key}: id {row['id']} trùng giữa các shard, cần upstream_id có tên shard")
                if why:
                    bdrop.append({"id": row["id"], "upstream_id": row.get("upstream_id"), "reason": why})
                    continue
                seen.add(row["id"])
                row["audio"] = os.path.relpath(row.pop("_path"), d).replace(os.sep, "/")
                brows.append(row)
            rows += brows
            dropped += bdrop
            if not args.limit:
                part.write_text(json.dumps({"rows": brows, "dropped": bdrop}, ensure_ascii=False), encoding="utf-8")
            print(f"  {key}: shard {bi + 1}/{len(batches)} xong ({len(brows)} câu)", flush=True)
            if not args.keep_source and spec["reader"] == "parquet":
                for f in files:
                    Path(f).unlink(missing_ok=True)
        with open(d / "rows.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta = {"set": key, "repo": spec["repo"], "revision": spec["revision"], "split": spec["split"],
                "files": [{"file": fn, "sha256": s} for fn, s in spec["files"]], "rows": len(rows),
                "expected": spec.get("expected"), "hours": round(sum(r["duration"] for r in rows) / 3600, 4),
                "speakers": len({r["speaker"] for r in rows}), "dropped": dropped, "limit": args.limit,
                "derivation": sorted({r["audio_derivation"] for r in rows}), "sec": round(time.time() - t0, 1)}
        (d / "ingest_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        flag = "" if args.limit or spec.get("expected") in (None, len(rows)) else f"  <-- KHÁC số công bố {spec['expected']}"
        print(f"{key}: {len(rows)} câu, {meta['hours']:.2f} h, {meta['speakers']} speaker, bỏ {len(dropped)}{flag}", flush=True)
        if not args.keep_source:
            shutil.rmtree(src, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
