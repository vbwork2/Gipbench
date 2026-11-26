"""Report VietNote research status or assemble the verified V03 training inputs."""

import argparse
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.paths import REPO
from gipbench.model.recipe import experiment as X
from gipbench.model.readiness import build_bundle, data_ready, model_validation, read_json, read_yaml, registry


def optional_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def quick_results(path):
    runs = []
    for subset in ("interim_cs_quick", "vi_quick"):
        for model in ("gip15_mbs4_open", "vietnote78_offline_int8", "vietnote78_streaming_int8_pad660",
                      "vietnote78_offline_fp32_t4", "vietnote78_streaming_fp32_t4_pad660"):
            result = optional_json(path / subset / model / "score/summary.json")
            score = result.get("profiles", {}).get("gipbench_strict_v1", {}).get("overall")
            if score and isinstance(score.get("wer"), (int, float)):
                runs.append((subset, model, score))
    return runs


def pct(value):
    return f"{value:.2f}%" if isinstance(value, (int, float)) else "—"


def full_baselines(path):
    runs = []
    for subset, ref in (("vi", "dev/dev_v2_vi@1"), ("cs_diagnostic", "dev/dev_interim_cs@1")):
        directory = path / subset / "vietnote78_offline_fp32_t4"
        summary = optional_json(directory / "score/summary.json")
        meta = optional_json(directory / "run_meta.json")
        if not summary or not meta:
            continue
        release = R.resolve(ref)
        digest = manifest_sha256(read_jsonl(release.manifest))
        hyps_digest = R.sha256(directory / "hyps.jsonl")
        score = summary["profiles"]["gipbench_strict_v1"]["overall"]
        if (summary.get("manifest_sha256") == meta.get("manifest_sha256") == digest
                and summary.get("hyps_sha256") == meta.get("hyps_sha256") == hyps_digest
                and summary.get("hyp_check", {}).get("ok") is True
                and score["utt"] == release.meta["files"]["manifest.jsonl"]["rows"]):
            runs.append((ref, score, meta["rtf"]))
    return runs


def int8_isolation(paths, parity):
    measured = optional_json(paths["int8_isolation"])
    if not measured or not parity:
        return {}
    try:
        expected = measured["outputs"]
        actual_paths = (paths["int8_isolation"].parent / "same_runtime").glob("*/*.jsonl")
        hashes = {file.relative_to(paths["int8_isolation"].parent / "same_runtime").as_posix(): R.sha256(file)
                  for file in actual_paths}
        if (hashes == expected and len(expected) == 10 and measured["rows"] == 113
                and measured["manifest_sha256"] == parity["dev_manifest_sha256"]
                and all(measured[key] == parity[key] for key in ("config_sha256", "catalog_sha256", "scorer_code_sha256"))
                and measured["parent_parity_sha256"] == R.sha256(paths["parity_before_softmax"])):
            return measured
    except (KeyError, OSError):
        pass
    return {}


def int8_table(measured, hypotheses, official, encoder_diff):
    runtime = measured["runtime"]
    counts = measured["platform_reference_comparison"]
    lines = ["", "### Đối chiếu INT8 cùng runtime trước khi sửa exporter", "",
             f"Cả bản chính thức và các biến thể đều được giải mã lại trên {runtime['platform']} với "
             f"Sherpa-ONNX {runtime['sherpa-onnx']}, beam 4 và 4 luồng CPU. Các graph lượng tử hóa lại dùng "
             f"ONNX Runtime {runtime['quantizer_onnxruntime']}. Dùng đúng {measured['rows']} câu đã khóa, "
             f"SHA256 manifest `{measured['manifest_sha256']}`.", "",
             f"Bản chính thức khớp kết quả tham chiếu lần trước ở "
             f"{counts['offline']['exact']}/{measured['rows']} câu offline và "
             f"{counts['streaming']['exact']}/{measured['rows']} câu streaming.", "",
             "| Chế độ | Biến thể | Khớp transcript | ΔWER (điểm %) | ΔCS-WER (điểm %) | Đạt |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    label = {"requantized_official_fp32": "Lượng tử hóa lại FP32 chính thức",
             "reexport_encoder": "Thay encoder", "reexport_decoder_joiner": "Thay decoder và joiner",
             "full_reexport": "Re-export toàn bộ"}
    for key, item in measured["results"].items():
        mode, variant = key.split("/")
        wer = item["wer"][1] - item["wer"][0]
        cs_wer = item["cs_wer"][1] - item["cs_wer"][0]
        lines.append(f"| {mode} | {label[variant]} | {item['exact']}/{item['rows']} | {wer:+.2f} | "
                     f"{cs_wer:+.2f} | {'Có' if item['passed'] else 'Chưa'} |")
    streaming = [item for key, item in measured["results"].items() if key.startswith("streaming/")]
    lines += ["", f"Các biến thể streaming được đo trên {streaming[0]['long_rows']} câu từ 20 giây. "
              f"Số câu mất lượt phát sau nửa câu nhiều nhất là {max(item['lost_late_emissions'] for item in streaming)}."]
    for mode in ("offline", "streaming"):
        encoder = read_jsonl(hypotheses / mode / "reexport_encoder.jsonl")
        full = read_jsonl(hypotheses / mode / "full_reexport.jsonl")
        exact = sum(a["id"] == b["id"] and a["hyp"] == b["hyp"] for a, b in zip(encoder, full, strict=True))
        lines.append(f"Ở chế độ {mode}, biến thể chỉ thay encoder khớp {exact}/{measured['rows']} transcript với bản re-export toàn bộ.")
    if all(measured["results"][f"{mode}/requantized_official_fp32"]["artifact_sha256"]["encoder.int8.onnx"]
           == official[f"{mode}/int8/encoder.int8.onnx"] for mode in ("offline", "streaming")):
        lines.append("Cả hai encoder lượng tử hóa lại từ FP32 chính thức đều trùng SHA256 với encoder INT8 phát hành.")
    for mode, item in encoder_diff.items():
        full = measured["results"].get(f"{mode}/full_reexport", {})
        if (item.get("published_encoder_sha256") != official.get(f"{mode}/int8/encoder.int8.onnx")
                or item.get("reexport_encoder_sha256") != full.get("artifact_sha256", {}).get("encoder.int8.onnx")):
            continue
        initializers = item["initializer_values_changed"]
        constants = item["constant_nodes_changed"]
        if not all(value["dtype"] == "float32" for value in (*initializers.values(), *constants.values())):
            continue
        lines.append(f"So graph {mode}, {item['qint8_initializers']} initializer INT8 và "
                     f"{item['float_scale_initializers']} scale FP32 có cùng giá trị. "
                     f"Có {len(initializers)} initializer FP32 và {len(constants)} nút Constant FP32 khác giá trị.")
    return lines


def encoder_constants(paths, isolation, official):
    measured = optional_json(paths["encoder_constants"])
    if not measured or not isolation:
        return {}
    root = paths["encoder_constants"].parent
    try:
        if (measured["rows"] != isolation["rows"] or measured["manifest_sha256"] != isolation["manifest_sha256"]
                or measured["matched_runtime_sha256"] != R.sha256(paths["int8_isolation"])
                or measured["graph_diff_sha256"] != R.sha256(paths["encoder_graph_diff"])
                or any(isolation["runtime"][key] != measured["runtime"][key]
                       for key in ("platform", "python", "sherpa-onnx"))
                or set(measured["results"]) != {"offline", "streaming"}):
            return {}
        for mode, cases in measured["results"].items():
            if set(cases) != {"downsampling_multiplier", "positional_encoding", "both"}:
                return {}
            for name, item in cases.items():
                encoder = "encoder.int8.onnx"
                digest = item["encoder_sha256"]
                if R.sha256(root / mode / name / encoder) != digest:
                    return {}
                if item["identical_encoder_control"] == "published":
                    if digest != official[f"{mode}/int8/{encoder}"]:
                        return {}
                elif item["identical_encoder_control"] == "reexport":
                    if digest != isolation["results"][f"{mode}/reexport_encoder"]["artifact_sha256"][encoder]:
                        return {}
                elif R.sha256(root / mode / f"{name}.jsonl") != item["hypotheses_sha256"]:
                    return {}
        return measured
    except (KeyError, OSError):
        return {}


def constants_table(measured):
    labels = {"downsampling_multiplier": "Hệ số nhân downsampling",
              "positional_encoding": "Bảng mã hóa vị trí", "both": "Cả hai nhóm"}
    lines = ["", "### Thử thay từng nhóm hằng FP32 trong encoder INT8", "",
             f"Mỗi biến thể chỉ lấy giá trị FP32 đã chọn từ bản phát hành, rồi so trên {measured['rows']} câu đã khóa. "
             "Giữ cố định decoder, joiner và cấu hình nhận dạng dùng trong phép hoán đổi encoder.", "",
             "| Chế độ | Nhóm được khôi phục | Khớp bản phát hành | Khớp encoder re-export |",
             "| --- | --- | ---: | ---: |"]
    for mode, cases in measured["results"].items():
        for name, item in cases.items():
            published = item["vs_published"]
            reexport = item["vs_reexport"]
            lines.append(f"| {mode} | {labels[name]} | {published['exact']}/{published['rows']} | "
                         f"{reexport['exact']}/{reexport['rows']} |")
    lines.append("")
    for mode, cases in measured["results"].items():
        exact = measured["rows"]
        for factor, other in (("downsampling_multiplier", "positional_encoding"),
                              ("positional_encoding", "downsampling_multiplier")):
            if (cases[factor]["vs_published"]["exact"] == exact
                    and cases[other]["vs_reexport"]["exact"] == exact):
                lines.append(f"Ở chế độ {mode}, khôi phục riêng nhóm {labels[factor].lower()} đưa toàn bộ transcript "
                             "về bản phát hành. Biến thể chỉ khôi phục nhóm còn lại cho nguyên transcript re-export.")
                break
        else:
            one_factor = [labels[name].lower() for name in labels if name != "both"
                          and cases[name]["vs_published"]["exact"] == exact]
            if one_factor:
                lines.append(f"Ở chế độ {mode}, khôi phục {', '.join(one_factor)} đủ để khớp toàn bộ transcript phát hành.")
            elif cases["both"]["vs_published"]["exact"] == exact:
                lines.append(f"Ở chế độ {mode}, cần khôi phục cả hai nhóm để khớp toàn bộ transcript phát hành trong các lượt đã đo.")
    if all(cases["both"]["identical_encoder_control"] == "published" for cases in measured["results"].values()):
        lines.append("Khôi phục cả hai nhóm tạo encoder trùng SHA256 bản phát hành ở cả hai chế độ. "
                     "Hai biến thể trùng byte dùng lại transcript đối chứng đã kiểm SHA256 trên cùng runtime.")
    lines.append("Các biến thể trên dùng để xác định nguyên nhân lệch của exporter cũ.")
    return lines


def curation_section(curation):
    if not curation:
        return []
    cv, fl, lb = curation.get("cv_other", {}), curation.get("fleurs_vi", {}), curation.get("librispeech_en", {})
    vv, vm, vs = curation.get("vivos_vi", {}), curation.get("vimedcss_cs", {}), curation.get("vss_cs", {})
    dh = curation.get("datahive_vi", {})
    lines = [
        "", "## Tuyển lọc dữ liệu tự động theo nguồn gốc", "",
        "Phân tách 5 nhóm nguồn gốc (`human_labeled`, `existing_supervised`, `pseudo_labeled`, "
        "`unlabeled_conversational`, `synthetic_speech`) theo `configs/dataset/vietnote_curation_policy.yaml`. "
        "Số giờ/dòng đã duyệt thủ công giữ nguyên `0` nếu chưa nghe kiểm; **không** xem đồng thuận giữa các "
        "teacher ASR là nhãn chuẩn (ground truth).", "",
        "| Tập dữ liệu | Nguồn gốc | Chuyển cloud | Tổng (câu / giờ) | Giữ lại (câu / giờ) | WER teacher | Rà soát người |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    if fl:
        lines.append(
            f"| `pool/vietnote_fleurs_vi@v2` | existing_supervised | Có (CC-BY-4.0) | "
            f"{fl['total_rows']:,} / {fl['total_hours']:.3f}h | spot {fl['sample_rows']} câu (0 fail, {fl['sample_outliers_min_wer_gt35']} ngoại lệ) | "
            f"VN med {100 * fl['sample_median_vietnote_wer']:.2f}% | {fl['human_reviewed_rows']} |"
        )
    if cv:
        lines.append(
            f"| `pool/cv_other@v3` | existing_supervised | Có (CC0-1.0) | "
            f"{cv['total_rows']:,} / {cv['total_hours']:.3f}h | {cv['curated_rows']:,} / {cv['curated_hours']:.3f}h (loại {cv['dropped_rows']}: {cv['leak_drops']} leak) | "
            f"pair med {100 * cv['median_teacher_pair_wer']:.2f}% | {cv['human_reviewed_rows']} |"
        )
    if lb:
        lines.append(
            f"| `pool/vietnote_librispeech_en@v2` | existing_supervised | Có (CC-BY-4.0) | "
            f"{lb['total_rows']:,} / {lb['total_hours']:.3f}h | spot {lb['sample_rows']} câu (0 fail, {lb['sample_outliers_min_wer_gt35']} ngoại lệ) | "
            f"VN med {100 * lb['sample_median_vietnote_wer']:.2f}% | {lb['human_reviewed_rows']} |"
        )
    if vv:
        lines.append(
            f"| `pool/vivos_train@v3` | existing_supervised | Nội bộ (NC-SA) | "
            f"{vv['total_rows']:,} / {vv['total_hours']:.3f}h | spot {vv['sample_rows']} câu (0 fail, {vv['sample_outliers_min_wer_gt35']} ngoại lệ) | "
            f"pair med {100 * vv['sample_median_pair_wer']:.2f}% | {vv['human_reviewed_rows']} |"
        )
    if vm:
        lines.append(
            f"| `pool/vietnote_vimedcss_cs@v1` | existing_supervised | Chờ quyền YT | "
            f"{vm['total_rows']:,} / {vm['total_hours']:.3f}h | spot {vm['sample_rows']} câu (0 fail, {vm['sample_outliers_min_wer_gt35']} ngoại lệ) | "
            f"pair med {100 * vm['sample_median_pair_wer']:.2f}% | {vm['human_reviewed_rows']} |"
        )
    if vs:
        lines.append(
            f"| `pool/vss_cs@v5` | pseudo_labeled | Chờ quyền ghi/cloud | "
            f"{vs['total_rows']:,} / {vs['total_hours']:.3f}h | Qwen={vs['qwen_cached_rows']}, FW={vs['fw_cached_rows']} ({vs['span_confirmed_min1_rows']} span ok) | "
            f"pair med {100 * vs['median_qwen_fw_pair_wer']:.2f}% | {vs['human_reviewed_rows']} |"
        )
    if dh:
        lines.append(
            f"| `datahive_vi` (5 cuộc train) | unlabeled_conversational -> pseudo | Có (CC-BY-4.0) | "
            f"{dh['vad_segments']} VAD / {dh['vad_hours']:.3f}h (từ {dh['raw_call_hours']:.3f}h thô) | "
            f"{dh['curated_rows']} / {dh['curated_hours']:.3f}h (gồm {dh['curated_en_span_rows']} câu `<en>` = {dh['curated_en_span_hours']:.3f}h; loại {dh['reserved_term_drops']} thuật ngữ khóa) | "
            f"pair med {100 * dh['median_teacher_pair_wer_curated']:.2f}% (gốc {100 * dh['median_teacher_pair_wer_all']:.2f}%) | {dh['human_reviewed_rows']} |"
        )
    return lines


def exploratory_section(work, research, kaggle_runs, paths):
    expl_cfg = research.get("exploratory_experiments", {})
    expl_runs = [r for r in kaggle_runs if r.get("experiment") == "expl-ft" and r.get("state") == "completed"]
    ve01_eval = optional_json(paths.get("ve01_evaluation", Path("__missing__")))
    if not expl_cfg and not expl_runs and not ve01_eval:
        return []
    lines = [
        "", "## Nhánh nghiên cứu thăm dò (V02A & VE01 — tách biệt V03)", "",
        "- **V02A** (`configs/model/experiments/vietnote_v02a_validation.yaml`): giao thức kiểm định huấn luyện thay thế "
        "trên dữ liệu có quyền hợp lệ (tổng hợp `tiny-diag` Arm A LR 0,0045 vs Arm B LR 0,0015, `icefall-memory` 200s FP32 "
        "và `diag-ft`), độc lập về mặt khoa học và **không** thay thế hay xóa bỏ V02 lịch sử (163 câu VSS) lẫn cổng V03.",
        "- Các thí nghiệm `VE01` chỉ dùng kết luận thăm dò (`allow_generalization_claims: false`), giữ nguyên quyền bản ghi, "
        "chuyển cloud và chống rò rỉ benchmark.",
    ]
    for r in expl_runs:
        res = optional_json(work / "kaggle" / r["run"] / "output/expl-ft.json")
        if not res:
            continue
        df = res["duration_filter"]
        exp = res["exposure"]
        src_str = ", ".join(f"{k}={v:.3f}h" for k, v in res.get("source_hours", {}).items())
        lines += [
            "", f"### Thí nghiệm thăm dò `VE01` trên Kaggle T4 (`{r['run']}`)", "",
            f"- Cấu hình: LR={res['base_lr']:g}, FP32, `max_duration={res['max_duration']}`, "
            f"`utt_duration=[{res['min_utt_duration']:g}, {res['max_utt_duration']:g}]s`, {res['num_epochs']} epoch, seed {res['seed']}.",
            f"- Dữ liệu vào: {df['before']['count']} câu ({df['before']['hours']:.3f}h: {src_str}); "
            f"giữ lại {df['after']['count']} câu ({df['after']['hours']:.3f}h, {df['after_gt_20s']} câu >20s); "
            f"{exp['steps']} bước, tiếp xúc {exp['total_audio_hours']:.3f}h ({exp['gt_20s_cut_views']} lượt >20s); "
            f"đỉnh VRAM {res.get('peak_allocated_mb', '—')} MB; val loss `{res.get('validation_losses')}`; chọn epoch {res['selected_epoch']}.",
            "", "| Epoch (Kaggle GPU greedy) | Bước | WER | CS-WER | Recall Anh | VI-only WER | S / D / I | >20s WER (D) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for sc in res.get("scores_by_epoch", []):
            b20 = sc.get("by_duration_band", {}).get(">20s")
            b20_str = f"{pct(b20['wer'])} (D={b20['D']})" if b20 else "—"
            tag = f"{sc['epoch']} *" if sc["epoch"] == res["selected_epoch"] else str(sc["epoch"])
            lines.append(
                f"| {tag} | {sc['steps']} | {pct(sc['wer'])} | {pct(sc.get('cs_wer'))} | "
                f"{pct(sc.get('term_recall'))} | {pct(sc.get('vi_only_wer'))} | "
                f"{sc['S']}/{sc['D']}/{sc['I']} | {b20_str} |"
            )
    if ve01_eval.get("models"):
        lines += [
            "", "### Đánh giá Sherpa-ONNX cục bộ (Windows, beam 4): `VE01` vs `pretrained`", "",
            "| Model | Tập chấm | Chế độ | WER | CS-WER | Recall Anh | VI-only WER | S / D / I | >20s WER (D) | Mất phát chữ >=20s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for ds in ("authorized_dev", "vi_quick", "interim_cs_quick", "dev_interim_cs", "dev_v2_vi_strat300"):
            for mode in ("offline_fp32", "offline_int8", "streaming_fp32", "streaming_int8"):
                for mname in ("pretrained", "ve01"):
                    sc = ((ve01_eval["models"].get(mname) or {}).get(ds) or {}).get(mode)
                    if not sc:
                        continue
                    b20 = sc.get("by_duration_band", {}).get(">20s")
                    b20_str = f"{pct(b20['wer'])} (D={b20['D']})" if b20 else "—"
                    sl = sc.get("streaming_long")
                    sl_str = f"{sl['lost_late_emissions']}/{sl['long_rows']}" if sl and sl["long_rows"] else "—"
                    lines.append(
                        f"| `{mname}` | `{ds}` | `{mode}` | {pct(sc['wer'])} | {pct(sc.get('cs_wer'))} | "
                        f"{pct(sc.get('term_recall'))} | {pct(sc.get('vi_only_wer'))} | "
                        f"{sc['S']}/{sc['D']}/{sc['I']} | {b20_str} | {sl_str} |"
                    )
        paired = ve01_eval.get("paired_vs_pretrained", {})
        if paired:
            lines += [
                "", "#### Bootstrap ghép cặp 95% CI (`VE01` − `pretrained`)", "",
                "| Tập chấm | Chế độ | Metric | Pretrained | VE01 | Δ | CI 95% |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
            for ds, modes in paired.items():
                for mode, metrics in modes.items():
                    for metric, item in metrics.items():
                        ci = f"[{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]" if item.get("ci95") else "—"
                        delta = f"{item['delta']:+.2f}" if isinstance(item.get("delta"), (int, float)) else "—"
                        lines.append(
                            f"| `{ds}` | `{mode}` | `{metric}` | {pct(item.get('a'))} | "
                            f"{pct(item.get('b'))} | {delta} | {ci} |"
                        )
    return lines


def artifact_metrics(paths, detail, quick):
    lines = []
    if quick:
        lines += ["| Dev nhanh | Model | WER | CS-WER | Recall Anh |", "| --- | --- | ---: | ---: | ---: |"]
        for subset, system, score in quick:
            lines.append(f"| {subset} | {system} | {pct(score['wer'])} | {pct(score.get('cs_wer'))} "
                         f"| {pct(score.get('term_recall'))} |")
        lines.append("")
    strict = optional_json(paths["strict_offline"])
    smoke = optional_json(paths["single_step"])
    if strict:
        lines.append(f"- PyTorch offline đã lưu {strict['exact']}/{strict['rows']} transcript trùng FP32.")
    if smoke:
        lines.append(f"- GPU smoke một bước có loss {smoke['loss']:.5f}, CUDA allocated đỉnh "
                     f"{smoke['peak_cuda_allocated_bytes'] / 2**30:.2f} GiB, checkpoint reload {smoke['reloaded']}.")
    for name, item in detail.get("parity", {}).get("comparisons", {}).items():
        lines.append(f"- {name} chính thức và re-export khớp {item['exact']}/{item['rows']} transcript. "
                     f"WER {pct(item['wer'][0])} → {pct(item['wer'][1])}, "
                     f"CS-WER {pct(item['cs_wer'][0])} → {pct(item['cs_wer'][1])}. "
                     f"Kết quả {'đạt' if item['passed'] else 'không đạt'}.")
        if "long_rows" in item:
            lines.append(f"  {item['long_rows']} câu dài, mất {item['lost_late_emissions']} lượt phát sau nửa câu. "
                         f"Độ lệch lớn nhất ở thời điểm phát chữ đầu tiên là {item['max_first_emission_shift_seconds']} giây.")
    tensors = detail.get("tensors", {})
    for mode, stats in tensors.get("measurements", {}).items():
        counts = sum(item["comparisons"] for item in stats.values())
        failed = sum(item["failures"] for item in stats.values())
        lines.append(f"- Tensor FP32 {mode} có {failed} phép ngoài dung sai trong {counts} phép.")
    if tensors.get("quantization"):
        quant = tensors["quantization"]
        lines.append(f"- Kiểm cấu trúc INT8 theo kênh đạt {sum(item['passed'] for item in quant.values())}/{len(quant)} graph.")
    for arm in ("tiny", "tiny_followup"):
        tiny = detail.get(arm)
        if not tiny:
            continue
        start, end = tiny["score_by_pass"][0], tiny["score_by_pass"][-1]
        lr = f"LR={tiny['lr']:g}" if "lr" in tiny else "LR=0.0045"
        lines.append(f"- Tiny-set {tiny['train_rows']} câu, {tiny['steps']} bước, {lr}. "
                     f"WER {pct(start['wer'])} → {pct(end['wer'])}, CS-WER {pct(start['cs_wer'])} → {pct(end['cs_wer'])}, "
                     f"recall Anh {pct(start['term_recall'])} → {pct(end['term_recall'])}. "
                     f"Đỉnh allocated {tiny['peak_allocated_bytes'] / 2**30:.2f} GiB. "
                     f"Kết quả {'đạt' if tiny['passed'] else 'không đạt'}.")
    attempt = optional_json(paths["tiny_followup_attempt"])
    if not detail.get("tiny_followup") and attempt.get("status") == "blocked_cuda":
        lines.append(f"- Lượt LR thấp dừng ở {attempt['completed_steps']} bước. "
                     f"Windows thấy {attempt['host_gpu']}, còn PyTorch trong WSL thấy "
                     f"{attempt['torch_cuda_device_count']} thiết bị CUDA và NVML báo lỗi.")
    for item in detail.get("memory", {}).get("cases", []):
        result = f"{len(item.get('steps', []))} bước, "
        if "peak_allocated_bytes" in item:
            result += (f"allocated {item['peak_allocated_bytes'] / 2**30:.2f} GiB, "
                       f"reserved {item['peak_reserved_bytes'] / 2**30:.2f} GiB, "
                       f"clip dài nhất {item['longest_clip']:.2f} giây, ")
        result += "đạt" if item["passed"] else item.get("reason", "thiếu 10% dự phòng").split(". ")[0]
        lines.append(f"- Batch {item.get('kind', 'probe')} {item['limit']} giây, {result}.")
    if paths["espeak"].is_file():
        lines.append(f"- eSpeak NG được lưu làm đối chứng âm tại {paths['espeak'].relative_to(REPO).as_posix()}.")
    neural = optional_json(paths["zerotts"])
    if neural:
        lines.append(f"- ZeroTTS sinh {neural['rows']} câu, {neural['hours']:.3f} giờ. "
                     f"Đã duyệt bởi người nghe {neural['human_approved_rows']} câu.")
    for name, score in optional_json(paths["zerotts_asr"]).items():
        lines.append(f"- TTS {name} qua ASR có WER {pct(score['wer'])}, CS-WER {pct(score['cs_wer'])}, "
                     f"recall Anh {pct(score['term_recall'])}.")
    review = paths["zerotts_asr"].parent
    triage = [review / f"{name}_triage.jsonl" for name in ("zerotts", "espeak")]
    if neural and (review / "blind_review.csv").is_file() and all(path.is_file() for path in triage):
        zero, control = (len(read_jsonl(path)) for path in triage)
        lines.append(f"- Lọc tự động giữ {zero}/{neural['rows']} mẫu ZeroTTS và {control}/{neural['rows']} mẫu eSpeak. "
                     "Phiếu nghe mù đã sẵn sàng. Chờ người duyệt.")
    vieneu = optional_json(paths["vieneu"])
    if vieneu:
        queue = paths["vieneu"].parent
        kept = queue / "vieneu_triage.jsonl"
        scores = optional_json(paths["vieneu_asr"])
        lines.append(f"- VieNeu-TTS CPU sinh {vieneu['rows']} câu ({vieneu['hours']:.4f} giờ). "
                     f"Hai hệ ASR giữ {len(read_jsonl(kept)) if kept.is_file() else '—'} câu qua lọc.")
        for name, score in scores.items():
            if name.startswith("vieneu/"):
                lines.append(f"  {name} có WER {pct(score['wer'])}, CS-WER {pct(score['cs_wer'])} "
                             f"và recall Anh {pct(score['term_recall'])}.")
        lines.append("- Phiếu nghe mù ở `data/work/vietnote/tts/vieneu_pilot/blind_review.csv`. "
                     "Còn chờ phê duyệt của người nghe.")
    return lines


def kaggle_progress(work, runs, pilot_evaluation):
    doctor = optional_json(work / "kaggle/doctor.json")
    authenticated = doctor.get("authentication") == "verified"
    lines = ["", "## Kaggle", "",
             f"- CLI {doctor.get('cli') or 'chưa xác minh'}. Xác thực {'đã xác minh' if authenticated else 'đang bị chặn'}."]
    if not authenticated:
        lines.append("- Tạo token tại [Kaggle API Settings](https://www.kaggle.com/settings/api) rồi lưu "
                     "vào `%USERPROFILE%\\.kaggle\\access_token`.")
    lines += ["", "| Lượt | Trạng thái | Phiên bản từ xa | GPU đã xác minh |",
              "| --- | --- | --- | --- |"]
    for run in runs:
        if run["state"] == "prepared" and not (run["experiment"] == "V02" and run.get("runtime_lock")
                                               and run.get("bundle", {}).get("members_manifest_sha256")):
            continue
        gpu = run.get("environment", {}).get("gpu", "—")
        name = run["run"]
        if run.get("url"):
            name = f"[{name}]({run['url']})"
        lines.append(f"| {name} | {run['state']} | {run.get('version', '—')} | {gpu} |")
    submitted = [r for r in runs if r.get("version")]
    complete = [r for r in submitted if r.get("state") == "completed"]
    lines += ["", f"Đã gửi {len(submitted)} lượt. Đã đồng bộ và xác minh {len(complete)} lượt. "
              "V02 cần văn bản quyền huấn luyện và chuyển đúng 163 bản ghi."]
    for run in submitted:
        if run["experiment"] == "icefall-pilot" and run["state"] == "failed":
            log = optional_json(work / "kaggle" / run["run"] / "logs.json").get("logs", "")
            if "The pinned dataset archive was unavailable or ambiguous" in log:
                lines.append(f"- {run['run']} dừng trước huấn luyện. Kaggle đã giải nén archive thành "
                             "`bundle/pilot.json`. Lượt khôi phục kiểm SHA256 payload trong thư mục mount.")
            elif "ModuleNotFoundError: No module named 'onnxruntime'" in log:
                lines.append(f"- {run['run']} tới bước export sau huấn luyện và resume. "
                             "Exporter dừng vì thiếu ONNX Runtime. Lượt khôi phục ghim wheel theo SHA256.")
    for run in sorted(complete, key=lambda item: item["experiment"] == "icefall-memory"):
        if run["experiment"] == "icefall":
            probe = optional_json(work / "kaggle" / run["run"] / "output/icefall.json")
            lines += ["", f"- Recipe VietNote/Icefall `finetune.py` trên {run['environment']['gpu']} "
                      f"đã chạy {probe['steps']} bước với audio tổng hợp và {probe['parameters']:,} tham số. "
                      f"Train loss mỗi frame {probe['train_loss']:.5f}.",
                      f"- Đã nạp lại checkpoint vào model mới và đối chiếu trọng số, optimizer và scheduler. "
                      f"SHA256 checkpoint `{probe['checkpoint_sha256']}`. Các wheel cài thêm được ghim trong "
                      "`requirements.txt`, phiên bản package của phiên chạy nằm trong `runtime-packages.json`."]
        if run["experiment"] == "icefall-memory":
            memory = optional_json(work / "kaggle" / run["run"] / "output/engineering-memory.json")
            steps = memory["steps"]
            headroom = (memory["gpu_free_before_bytes"] - memory["peak_reserved_bytes"]) / 2**30
            lines += ["", "### Probe bộ nhớ FP32 200 giây trên FLEURS", "",
                      f"- Tesla T4, `{memory['release']}`, câu dài nhất {memory['longest_clip']:.2f} giây. "
                      f"Hoàn tất {len(steps)}/{memory['planned_steps']} bước cập nhật.",
                      f"- Đỉnh CUDA allocated {memory['peak_allocated_bytes'] / 2**30:.2f} GiB và reserved "
                      f"{memory['peak_reserved_bytes'] / 2**30:.2f} GiB. "
                      f"VRAM trống trước probe {memory['gpu_free_before_bytes'] / 2**30:.2f} GiB. "
                      f"Dư địa ước tính tại đỉnh reserved {headroom:.2f} GiB."]
            if steps:
                lines += ["", "| Bước | Audio thật (giây) | Sau padding (giây) | Tensor đầu vào | Loss |",
                          "| ---: | ---: | ---: | --- | ---: |"]
                for step in steps:
                    shape = " × ".join(map(str, step["input_shape"]))
                    lines.append(f"| {step['step']} | {step['audio_seconds']:.2f} | {step['padded_seconds']:.2f} "
                                 f"| {shape} | {step['loss']:.5f} |")
            if memory.get("reason"):
                lines += ["", f"Lỗi đo được là `{memory['reason']}`."]
            outcome = "Đạt ngưỡng kỹ thuật" if memory["passed"] else "Chưa đạt ngưỡng kỹ thuật"
            lines += ["", f"{outcome}. Probe gọi trực tiếp bước huấn luyện để đo batch dài bằng FLEURS. "
                      "Cổng V03 vẫn chưa đạt.",
                      f"[Artifact phép đo](../data/work/vietnote/kaggle/{run['run']}/output/engineering-memory.json)"]
        if run["experiment"] == "icefall-pilot":
            pilot = optional_json(work / "kaggle" / run["run"] / "output/pilot.json")
            equal = ", ".join(key for key, same in pilot["epoch_boundary_equal"].items() if same) or "không có"
            different = ", ".join(key for key, same in pilot["epoch_boundary_equal"].items() if not same)
            lines += ["", f"- Pilot FLEURS dùng {pilot['train_cuts']} câu train và {pilot['valid_cuts']} câu validation. "
                      f"Số bước trước và sau resume là {pilot['steps_before_resume']} và {pilot['steps_resumed']}. "
                      f"Loss validation theo epoch là {', '.join(f'{loss:.4f}' for loss in pilot['validation_losses'])}.",
                      f"- Trạng thái trùng lượt chạy liên tục tại cuối epoch 2 gồm {equal}. "
                      f"RNG {'có' if pilot['rng_saved'] else 'vắng'} trong checkpoint. "
                      f"Kết quả kiểm khôi phục áp dụng cho ranh giới epoch của pilot.",
                      f"- Checkpoint và ba graph ONNX FP32 offline đã đồng bộ qua manifest. "
                      f"SHA256 checkpoint `{pilot['checkpoint_sha256']}`."]
            if different:
                lines.append(f"- Lượt resume lệch lượt liên tục ở {different}. Kiểm phục hồi chính xác chưa đạt.")
            if (pilot_evaluation.get("run") == run["run"]
                    and pilot_evaluation.get("checkpoint_sha256") == pilot["checkpoint_sha256"]
                    and pilot_evaluation.get("result_manifest_sha256") == run["result_manifest_sha256"]):
                base, exported = pilot_evaluation["pretrained"], pilot_evaluation["exported"]
                wer = next(metric for metric in pilot_evaluation["paired"] if metric["metric"] == "wer")
                lines.append(f"- Chấm offline FP32 trên {exported['utt']} câu validation FLEURS "
                             f"cho WER {pct(base['wer'])} → {pct(exported['wer'])} và "
                             f"CER {pct(base['cer'])} → {pct(exported['cer'])}. "
                             f"ΔWER {wer['delta']:+.2f} điểm phần trăm, "
                             f"CI 95% ghép cặp [{wer['ci95'][0]:+.2f}, {wer['ci95'][1]:+.2f}]. "
                             "Pilot tiếng Việt đọc chưa cải thiện.")
                lines.append(f"- Log T4 ghi đỉnh allocated {pilot_evaluation['peak_allocated_mb']:,} MB "
                             f"với giới hạn {pilot_evaluation['pilot_max_duration']} giây/batch. V03 cần probe 200 giây.")
        if run["experiment"] == "synthetic":
            probe = optional_json(work / "kaggle" / run["run"] / "output/probe.json")
            lines += ["", f"- Lhotse {probe['lhotse']} trích {probe['features'][0]} khung × "
                      f"{probe['features'][1]} hệ số log-Mel từ âm thanh tổng hợp 1 giây, 16 kHz.",
                      f"- k2 {probe['k2']} chạy RNN-T trên {run['environment']['gpu']}. "
                      f"Loss {probe['loss']:.5f}, chuẩn L2 gradient của trọng số phép chiếu "
                      f"{probe['gradient_norm']:.5f}. Giới hạn phiên {run['timeout_seconds']} giây.",
                      f"- SHA256 manifest kết quả `{run['result_manifest_sha256']}`."]
        if run["experiment"] == "smoke":
            env = run["environment"]
            lines += ["", f"- Smoke v{run['version']} đạt phép nhân ma trận CUDA trên {env['gpu']}. "
                      f"PyTorch thấy {env['gpu_count']} GPU, VRAM trống "
                      f"{env['gpu_free_bytes'] / 2**30:.2f}/{env['gpu_total_bytes'] / 2**30:.2f} GiB. "
                      f"Giới hạn phiên {run['timeout_seconds']} giây.",
                      f"- Python {env['python']}, PyTorch {env['torch']}, CUDA {env['cuda']}. "
                      f"k2 {env['packages'].get('k2') or 'chưa cài'}, "
                      f"lhotse {env['packages'].get('lhotse') or 'chưa cài'}.",
                      f"- SHA256 manifest kết quả `{run['result_manifest_sha256']}`."]
    return lines


def legacy_eval_aliases():
    try:
        old = R.resolve("bench/external-v1.0")
        current = R.resolve("bench/external-v1.1")
    except (OSError, R.ReleaseError):
        return 0
    if current.meta.get("flags", {}).get("derived_from") != old.ref:
        return 0
    verified = {row["id"]: row for row in read_jsonl(current.manifest)}
    fields = ("dataset", "upstream_id", "text_tagged")
    count = 0
    for row in read_jsonl(old.manifest):
        other = verified.get(row["id"], {})
        if (not row.get("audio") and other.get("audio_sha256") and Path(other["_audio_path"]).is_file()
                and all(row.get(key) == other.get(key) for key in fields)):
            count += 1
    return count


def report(config, research):
    paths = {key: REPO / path for key, path in research["evidence"].items()}
    corpus, data_issues = data_ready(config, research)
    try:
        detail, issues = model_validation(config, research)
    except (OSError, ValueError) as exc:
        detail, issues = {}, {"checkpoint": str(exc)}
    quick = quick_results(paths["quick_baseline"])
    full = full_baselines(paths["full_baseline"])
    medical = optional_json(paths["medical_candidates"])
    diagnostic = optional_json(paths["diagnostic_dev"])
    historical = optional_json(paths["parity_before_softmax"])
    isolation = int8_isolation(paths, historical)
    historical_official = historical.get("official_files", {})
    constants = encoder_constants(paths, isolation, historical_official)
    v01 = all(key in detail and key not in issues for key in ("parity", "tensors")) and "export" not in issues
    v02 = all(key in detail and key not in issues for key in ("tiny", "memory"))
    work = paths["single_step"].parents[1]
    kaggle_runs = [optional_json(file) for file in sorted((work / "kaggle").glob("*/run.json"))]
    icefall_completed = any(run.get("experiment") == "icefall" and run.get("state") == "completed"
                            for run in kaggle_runs)
    statuses = {
        "V00": f"Có {len(quick)} lượt quick baseline và {len(full)}/2 bộ dev đầy đủ đã chấm.",
        "V01": "Đạt." if v01 else "Một phần. Xem đối chiếu FP32, INT8 và cổng bên dưới.",
        "V02": "Đạt." if v02 else "Một phần. Xem tiny-set và batch FP32 bên dưới.",
        "V02P": ("CLI Icefall đã xác minh trên Kaggle. Pilot cục bộ có kết quả chấm riêng." if icefall_completed
                 else "Pilot cục bộ đã chạy. Chờ xác minh Kaggle." if optional_json(paths["cli_pilot"])
                 else "Chờ chạy pilot Icefall."),
        "V03": "Bị chặn bởi cổng dữ liệu hoặc model." if data_issues or not v01 or not v02 else
               "Các tiền đề dữ liệu và model đạt. Chờ bundle và kết quả.",
    }
    lines = ["# Trạng thái nghiên cứu VietNote", "",
             "Mục tiêu giảm lỗi chuyển mã Việt–Anh với thuật ngữ chưa thấy và giữ độ chính xác tiếng Việt.", "",
             "| Lượt | Trạng thái | Điều kiện trước |", "| --- | --- | --- |"]
    for name, item in research["experiments"].items():
        lines.append(f"| {name} | {statuses.get(name, 'Chờ kết quả V03 và điều kiện trước.')} "
                     f"| {', '.join(item['requires']) or '—'} |")
    lines += ["", "## Dữ liệu V03", "", "| Nhóm | Giờ có sẵn | Giờ đủ bằng chứng | Mục tiêu | Giờ đủ chuẩn còn thiếu |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for kind, target in config["research"]["target_unique_hours"].items():
        available, eligible = corpus["available"][kind], corpus["eligible"][kind]
        lines.append(f"| {kind.upper()} | {available:.3f} | {eligible:.3f} | {target:g} | {max(0, target - eligible):.3f} |")
    lines += ["", "Giờ có sẵn dùng audio tại chỗ, khử trùng theo SHA256 ghi trong manifest hoặc đường dẫn. "
              "Giờ đủ bằng chứng cần hồ sơ quyền sử dụng, kiểm nhãn, phân tách người nói và kiểm âm học của release.", "",
              "| Release trong hỗn hợp | Số dòng | Giờ có sẵn | Giờ đủ bằng chứng |",
              "| --- | ---: | ---: | ---: |"]
    for ref, item in corpus["releases"].items():
        lines.append(f"| {ref} | {item['rows']} | {item['available_hours']:.3f} | {item['eligible_hours']:.3f} |")
    lines += ["", "### Tiến độ lấy dữ liệu", "",
              "| Nguồn | Trạng thái registry | Giờ ứng viên cục bộ | Giờ qua audit kỹ thuật | Giờ duyệt tại repo | Giờ đủ V03 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    sources = read_yaml(REPO / research["sources"])["sources"]
    acquisitions, screens = {}, {}
    for name, item in sources.items():
        acquisition = optional_json(work / "pilots" / name / "acquisition.json")
        screened = optional_json(work / "pilots" / name / "screened.json")
        acquisitions[name], screens[name] = acquisition, screened
        hours = acquisition.get("candidate_hours")
        if item.get("local_pool"):
            try:
                hours = R.resolve(item["local_pool"]).meta["stats"]["hours"]
            except (R.ReleaseError, OSError, KeyError):
                pass
        refs = {item.get("local_pool"), *[r for r in corpus["releases"] if name in r.lower()
                                        or corpus["releases"][r]["source"] == name]}
        eligible = sum(corpus["releases"].get(ref, {}).get("eligible_hours", 0) for ref in refs)
        local = f"{hours:.3f}" if isinstance(hours, (int, float)) else "—"
        checked = screened.get("audited_unique_hours")
        checked = f"{checked:.3f}" if isinstance(checked, (int, float)) else "—"
        reviewed = screened.get("human_reviewed_hours", acquisition.get("human_reviewed_hours", 0.0))
        if name == "vss":
            reviewed = max(reviewed, detail.get("tiny", {}).get("train_hours", 0))
        lines.append(f"| {name} | {item['status']} | {local} | {checked} | {reviewed:.3f} | {eligible:.3f} |")
    supplementary = {}
    for source, ref in (("fleurs_vi", "pool/vietnote_fleurs_vi@v2"), ("librispeech", "pool/vietnote_librispeech_en@v2")):
        if (R.release_dir(ref) / "release.json").is_file():
            release = R.resolve(ref)
            supplementary[source] = sum(row["duration"] for row in read_jsonl(release.manifest)) / 3600
            lines.append(f"| {ref} | Release bổ trợ đã xác minh | — | {supplementary[source]:.3f} | "
                         f"{screens[source].get('human_reviewed_hours', 0):.3f} | 0.000 |")
    coverage = [screen["acoustic"] for screen in screens.values() if "acoustic" in screen]
    if coverage:
        references = max(item["n_bench_unique_audio"] for item in coverage)
        missing = max(item["n_bench_unreadable"] for item in coverage)
        lines += ["", f"Audit dấu vân tay đối chiếu tới {references:,} đường dẫn audio đánh giá."]
        if missing and missing == legacy_eval_aliases():
            lines.append(f"{missing:,} dòng E0 v1.0 khóa bằng metadata khớp ID, nguồn upstream và transcript "
                         "với bản E0 v1.1 có audio đã được đưa vào audit.")
        elif missing:
            lines.append(f"{missing:,} dòng benchmark hoặc dev thiếu audio cục bộ và đã được kiểm metadata.")
    zerospeech = acquisitions["zerospeech"]
    neural_hours = sum(optional_json(paths[name]).get("hours", 0) for name in ("zerotts", "vieneu"))
    novel_real = supplementary.get("fleurs_vi", 0.0)
    current_corpus = optional_json(work / "corpus_current.json").get("released_union", {})
    speech_groups = current_corpus.get("hours_by_speech_group", {})
    lines += ["", "### Kho bổ trợ theo loại dữ liệu", "",
              "| Loại | Mục tiêu giờ | Giờ ứng viên đo được | Giờ duyệt tại repo | Giờ qua tiêu chí của nhóm | Giờ còn thiếu đến mức tối thiểu |",
              "| --- | ---: | ---: | ---: | ---: | ---: |",
              f"| Giọng thật tiếng Việt và CS trong kho mở rộng | 50–100 | "
              f"{acquisitions['fleurs_vi'].get('candidate_hours', 0):.3f} từ đợt mới | "
              f"{screens['fleurs_vi'].get('human_reviewed_hours', 0):.3f} đợt mới | {novel_real:.3f} bổ trợ đã phát hành | "
              f"{max(0, 50 - novel_real):.3f} tính trên phần đã xác minh |",
              "| Tiếng Anh giọng Việt | 2–5 | 0.000 | 0.000 | 0.000 | 2.000 |",
              f"| Tiếng Anh bản ngữ | — | {acquisitions['librispeech'].get('candidate_hours', 0):.3f} | "
              f"{screens['librispeech'].get('human_reviewed_hours', 0):.3f} | {supplementary.get('librispeech', 0):.3f} | — |",
              f"| CS tổng hợp bên ngoài | 5 | {zerospeech.get('candidate_hours', 0):.3f} | "
              f"{zerospeech.get('human_reviewed_hours', 0):.3f} | 0.000 | 5.000 |",
              f"| TTS tùy chỉnh | 1–2 | {neural_hours:.4f} | 0.000 | 0.000 | 1.000 |",
              "", "FLEURS là tiếng Việt đọc. LibriSpeech là tiếng Anh bản ngữ. "
              "Cột giờ phát hành của kho mở rộng hiện ghi phần đợt mới, tránh cộng trùng v1 với v2."]
    if current_corpus:
        lines += [f"Kho train cục bộ sau gộp có {current_corpus['unique_audio_rows']:,} bản ghi duy nhất và "
                  f"{current_corpus['unique_audio_hours']:.3f} giờ theo SHA256 hoặc đường dẫn. "
                  f"Trong đó tiếng Anh bản ngữ {speech_groups.get('native_en', 0):.3f} giờ, "
                  f"tiếng Anh giọng Việt {speech_groups.get('accented_en', 0):.3f} giờ, "
                  f"âm thanh tổng hợp {speech_groups.get('synthetic', 0):.3f} giờ "
                  f"và các bản ghi khác {speech_groups.get('other_released_speech', 0):.3f} giờ."]
    lines += ["", "### Quyền truy cập và xét duyệt", "",
              "- VietSpeech và viVoice có cổng điều khoản trên Hugging Face và cần xác minh quyền bản ghi theo subset. "
              "[VietSpeech](https://huggingface.co/datasets/NhutP/VietSpeech), [viVoice](https://huggingface.co/datasets/capleaf/viVoice).",
              "- VietSuperSpeech, code_switch_OM, LSVSC và ViMedCSS cần hồ sơ quyền bản ghi và quyền chuyển lên Kaggle "
              "cho dữ liệu dùng từ xa. V02 có 163 câu VietSuperSpeech đã nghe duyệt, tương đương 0,601 giờ.",
              "- Common Voice VI tái sử dụng bản cục bộ. Bản EN cần thông tin truy cập Mozilla Data Collective. "
              "L2-ARCTIC cần biểu mẫu của PSI Lab hoặc quyền mirror KoelLabs, chỉ chọn một bản. "
              "[Common Voice](https://commonvoice.mozilla.org/vi/datasets), [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/).",
              "- CanVEC cần tác giả cấp audio. ZeroSpeech còn chờ quyền văn bản nguồn và nghe duyệt. "
              "VietPrism chờ yêu cầu truy cập riêng. [CanVEC](https://github.com/Bak3rLi/CanVEC)."]
    source_review = optional_json(paths["source_review"])
    cards = optional_json(paths["source_cards"])
    gated = {name: optional_json(paths["gated_sources"] / name / "metadata.json")
             for name in ("vietspeech", "vivoice")}
    reviews = optional_json(paths["execution_review"])
    datahive = optional_json(paths["datahive_acquired"])
    if source_review.get("local_queue"):
        queues = source_review["local_queue"]
        lines += ["", f"Hàng đợi kiểm nguồn có {queues['vss']['queued_rows']} câu VietSuperSpeech "
                  f"({queues['vss']['queued_hours']:.3f} giờ) và {queues['lsvsc']['queued_rows']} câu LSVSC "
                  f"({queues['lsvsc']['queued_hours']:.3f} giờ). Chưa bổ sung giờ đủ điều kiện V03."]
        om = queues.get("code_switch_om")
        if om:
            lines.append(f"Shard code_switch_OM đã có {om['unique_recordings']:,} clip duy nhất "
                         f"({om['acquired_hours']:.3f} giờ) và {om['queued_rows']} video trong phiếu kiểm nguồn.")
    if cards.get("cards", {}).get("vietspeech", {}).get("gated"):
        lines.append("Metadata theo revision xác nhận VietSpeech yêu cầu mở quyền trên Hugging Face.")
    if all(item.get("authenticated_access") for item in gated.values()):
        lines.append("Tài khoản đã xác minh quyền đọc thực tế (HTTP 206 PAR1) với VietSpeech và viVoice. "
                     "Thẻ VietSpeech chỉ liệt kê audio và transcription (không có ID người nói/nguồn ghi trong shard). "
                     "Thẻ viVoice có kênh nguồn YouTube (gồm cả kênh sách nói/thuyết minh như `@VoizFM`, `@FonosVietnam` — "
                     "không tính giọng đọc là hội thoại tự nhiên) và giới hạn nghiên cứu CC BY-NC-SA 4.0. "
                     "Cần bằng chứng quyền bản ghi, tách người nói và duyệt thủ công trước khi đưa vào V03.")
        shard = optional_json(paths["gated_sources"] / "vietspeech/shard_schema.json")
        if shard.get("revision") == gated["vietspeech"]["revision"]:
            lines.append(f"Footer shard train VietSpeech có {shard['shard_rows']:,} dòng và các trường "
                         f"`{', '.join(shard['fields'])}`. Cần hồ sơ ngoài shard để kiểm nguồn và tách người nói.")
    if datahive:
        lines += [f"Đã tải và kiểm hash {datahive['candidate_rows']} cuộc hội thoại tiếng Việt DataHive "
                  f"({datahive['candidate_hours']:.3f} giờ). Nguồn công bố quyền huấn luyện và CC BY 4.0. "
                  f"Cần phiên âm và đối chiếu người nói trước khi phát hành. "
                  f"[Điều khoản và đồng thuận]({datahive['recording_rights_evidence']})."]
    if reviews:
        lines.append(f"Phiếu VSS ưu tiên {reviews['vss']['known_domain_rows']} câu có domain cụ thể. "
                     f"Phiếu LSVSC có {reviews['lsvsc']['rows']} câu và cần phân biệt hội thoại với giọng đọc.")
    lines += ["", "### Dev CS độc lập", "",
              f"Y khoa còn {medical.get('filter', {}).get('n_keep', '—')} ứng viên qua lọc ban đầu, "
              f"{medical.get('speakers', '—')} người nói xác minh theo báo cáo ứng viên. "
              f"Release độc lập cần {research['independent_cs']['min_rows']} câu, "
              f"{research['independent_cs']['min_domains']} domain và "
              f"{research['independent_cs']['min_verified_speakers']} người nói đã xác minh.",
              f"Quick CS ({diagnostic.get('rows', '—')} câu) dùng chẩn đoán kỹ thuật. "
              "Cấm dùng cho kết luận tổng quát hóa hoặc chấp nhận V03."]
    if reviews.get("datahive_vi"):
        heldout = reviews["datahive_vi"]
        lines.append(f"Đã dành nguyên {len(heldout['diagnostic_source_calls'])} cuộc gọi DataHive cho ứng viên dev. "
                     f"Phiếu nghe có {heldout['diagnostic_candidates']} đoạn chẩn đoán. "
                     f"Đã duyệt {heldout['human_reviewed_rows']} đoạn và xác minh {heldout['verified_speakers']} người nói.")
        screening = optional_json(paths["execution_review"].parent / "datahive_vi/acoustic_train_dev.json")
        stats = screening.get("stats", {})
        if screening.get("queue_sha256") == heldout["queue_sha256"] and stats.get("n_bench_unique_audio"):
            lines.append(f"Đối chiếu dấu vân tay {stats['n_train_fp']} đoạn với "
                         f"{stats['n_bench_unique_audio']:,} tệp train/dev đã có cache, "
                         f"ghi nhận {stats['n_matches']} cặp khớp. Chờ audit đầy đủ khi có transcript đã duyệt.")
    lines += ["", "## Kết quả", ""]
    if full:
        lines += ["| Dev đầy đủ | Số câu | WER | CER | CS-WER | RTF CPU |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for ref, score, rtf in full:
            lines.append(f"| {ref} | {score['utt']:,} | {pct(score['wer'])} | {pct(score['cer'])} "
                         f"| {pct(score['cs_wer'])} | {rtf:.4f} |")
        lines += ["", "Baseline là VietNote offline FP32 gốc với beam 4, batch 1 và 4 luồng. "
                  "Bộ CS tạm dùng chẩn đoán.", ""]
    current = detail.get("parity", {})
    exporter = optional_json(paths["checkpoint_export"])
    weights = exporter.get("checkpoint_softmax")
    if weights and current.get("runtime") and "export" not in issues:
        runtime = current["runtime"]
        lines += [f"Exporter tính softmax FP32 của `{weights['key']}` trên PyTorch {weights['torch']} "
                  f"({weights['platform']}) từ checkpoint gốc và ghi hệ số vào encoder trước khi lượng tử hóa. "
                  f"Bản phát hành và bản export được giải mã trên "
                  f"{runtime['platform']}, Python {runtime['python']}, Sherpa-ONNX {runtime['sherpa-onnx']}. "
                  f"SHA256 bộ câu `{current['dev_manifest_sha256']}`.", ""]
    lines += artifact_metrics(paths, detail, quick)
    if reviews.get("tts"):
        lines.append(f"- Phiếu chung đã xáo trộn {reviews['tts']['rows']} mẫu ZeroTTS và VieNeu-TTS. "
                     f"Đã duyệt {reviews['tts']['human_reviewed_rows']} mẫu.")
    if zerospeech.get("cache_only"):
        lines.append(f"- Tiếp tục ZeroSpeech từ cache. Có {zerospeech['candidate_rows']} ứng viên, "
                     f"{zerospeech['candidate_hours']:.3f} giờ và {zerospeech['new_download_bytes']} byte tải mới.")
    pilot = optional_json(paths["cli_pilot"])
    if pilot and pilot.get("selection", {}).get("scores"):
        first, last = pilot["selection"]["scores"][0], pilot["selection"]["scores"][-1]
        lines += [f"- Pilot Icefall cục bộ trên {last['utt']} câu dev có WER {pct(first['wer'])} → {pct(last['wer'])} "
                  f"và CS-WER {pct(first['cs_wer'])} → {pct(last['cs_wer'])}. "
                  f"Kết quả tổng {'đạt' if pilot['passed'] else 'không đạt'}."]
    if isolation:
        lines += int8_table(isolation, paths["int8_isolation"].parent / "same_runtime", historical_official,
                            optional_json(paths["encoder_graph_diff"]))
    if constants:
        lines += constants_table(constants)
        probe = optional_json(paths["encoder_constants"].with_name("softmax_probe.json"))
        reexport = isolation["results"]["offline/reexport_encoder"]["artifact_sha256"]["encoder.int8.onnx"]
        if (probe.get("checkpoint_sha256") == historical["checkpoint_input_sha256"]
                and probe.get("published_encoder_sha256") == historical_official["offline/int8/encoder.int8.onnx"]
                and probe.get("reexport_encoder_sha256") == reexport):
            for key, methods in probe["checkpoint_softmax"].items():
                if methods["torch_softmax_fp32"]["equal_published"]:
                    delta = probe["max_abs_delta"]
                    lines += ["", f"Softmax FP32 của `{key}` từ checkpoint gốc trên PyTorch {probe['torch']} "
                              f"khớp bit với hệ số phát hành. Bản re-export lệch cả {probe['different_entries']} phần tử, "
                              f"với độ lệch tuyệt đối tối đa {delta:.3g}."]
            before_quantization = probe.get("before_quantization", {})
            if set(before_quantization) == {"offline", "streaming"} and all(
                    item["published_fp32_matches_published_int8"] and item["reexport_fp32_matches_reexport_int8"]
                    for item in before_quantization.values()):
                lines.append("Các hệ số đã khác nhau giữa bản phát hành và bản re-export trước khi lượng tử hóa "
                             "ở cả chế độ offline và streaming.")
    pilot_evaluation = optional_json(paths["icefall_pilot_evaluation"])
    lines += kaggle_progress(work, kaggle_runs, pilot_evaluation)
    duration_audit = optional_json(paths.get("duration_audit", Path("__missing__")))
    if duration_audit.get("datasets"):
        lines += ["", "## Khảo sát phân bố thời lượng", "",
                  f"- Vị trí bộ lọc cứng trong recipe Icefall ghim: `{duration_audit['recipe_filter_location']}` "
                  f"(`1.0s <= duration <= 20.0s` và `T >= len(tokens)`).",
                  f"- Cấu hình `vietnote_v03.yaml` đặt `features.max_duration = {duration_audit['v03_mixture_configured_max_duration_seconds']:g}s`, "
                  "tạo sai lệch ngầm nếu không cấu hình tường minh qua `vietnote_logged_cli.py --min-utt-duration/--max-utt-duration` "
                  "(mặc định vẫn giữ `1.0–20.0s` làm đối chứng upstream).",
                  f"- Lưu ý augment: {duration_audit['speed_perturb_note']}",
                  "- Không kết luận sai lệch thời lượng gây ra thất bại X0 khi chưa có bằng chứng thực nghiệm đối chứng.", "",
                  "| Tập dữ liệu | Vai trò | Số câu | Giờ | P50 (s) | P90 (s) | Max (s) | >20s (câu / giờ) | Giữ lại (1–20s) |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, item in duration_audit["datasets"].items():
            p20 = item["policies"]["1.0-20s"]
            lines.append(
                f"| `{name}` | {item['role']} | {item['rows']:,} | {item['hours']:.3f} | "
                f"{item['p50']:.2f} | {item['p90']:.2f} | {item['max']:.2f} | "
                f"{item['gt_20s_rows']} ({item['gt_20s_hours']:.3f}h) | "
                f"{p20['kept_rows']:,} ({p20['retention_hour_pct']:.1f}% giờ) |"
            )
    tiny_diags = [r for r in kaggle_runs if r.get("experiment") == "tiny-diag" and r.get("state") == "completed"]
    if tiny_diags:
        lines += ["", "## Chẩn đoán LR trên tập nhỏ được cấp phép (không phải V02)", "",
                  "Do 163 bản ghi VietSuperSpeech của V02 vẫn thiếu văn bản quyền bản ghi/chuyển cloud, "
                  "thí nghiệm đối chứng LR độc lập này dùng 163 câu được cấp phép CC-BY-4.0 / CC0-1.0 "
                  "(FLEURS VI + Common Voice VI + LibriSpeech EN, mỗi câu 8,1–16,0s, 489 bước FP32, seed 0 trên Tesla T4). "
                  "Kết quả **không** thay thế cổng V02 đã đăng ký.", "",
                  "| Nhánh | LR | Bước | Tiếp xúc (giờ) | WER đầu → cuối | CS-WER đầu → cuối | Recall Anh đầu → cuối | S / D / I cuối | Loss đầu → cuối | Grad L2 (med / max) | Peak VRAM (GiB) |",
                  "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: |"]
        for r in sorted(tiny_diags, key=lambda x: x.get("arm", "")):
            res = optional_json(work / "kaggle" / r["run"] / "output/result.json")
            if not res:
                continue
            s0, s1 = res["score_by_pass"][0], res["score_by_pass"][-1]
            ts = res.get("training_summary", {})
            lines.append(
                f"| Arm {res['arm']} (`{r['run']}`) | {res['lr']:g} | {res['steps']} | "
                f"{ts.get('total_audio_hours', 0):.3f} | "
                f"{pct(s0['wer'])} → {pct(s1['wer'])} | {pct(s0.get('cs_wer'))} → {pct(s1.get('cs_wer'))} | "
                f"{pct(s0.get('term_recall'))} → {pct(s1.get('term_recall'))} | "
                f"{s1.get('S', '—')}/{s1.get('D', '—')}/{s1.get('I', '—')} | "
                f"{ts.get('loss_first', 0):.1f} → {ts.get('loss_last', 0):.1f} | "
                f"{ts.get('grad_l2_median', 0):.1f} / {ts.get('grad_l2_max', 0):.1f} | "
                f"{res['peak_allocated_bytes'] / 2**30:.2f} |"
            )
    diag_fts = [r for r in kaggle_runs if r.get("experiment") == "diag-ft" and r.get("state") == "completed"]
    if diag_fts:
        lines += ["", "## Fine-tune chẩn đoán đầu-cuối & đối chứng chính sách thời lượng (dữ liệu cấp phép)", "",
                  "Huấn luyện chính thức bằng Icefall `finetune.py` + `vietnote_logged_cli.py` trên hỗn hợp cấp phép "
                  "CC-BY-4.0 / CC0-1.0 (FLEURS VI gồm 35 câu >20s + Common Voice VI + LibriSpeech EN, 3 epoch, FP32, "
                  "LR 0,0015, seed 0) với tập dev tách hoàn toàn nguồn và người nói (71 câu, gồm 14 câu >20s). "
                  "Thí nghiệm này chỉ mang tính chẩn đoán kỹ thuật, **không** dùng dữ liệu test khóa và **không** suy rộng cho V03.", "",
                  "| Chính sách | Lượt Kaggle | Câu giữ lại (>20s) | Bước / Giờ tiếp xúc | Epoch chọn | Dev WER (ep0 → chọn) | Dev >20s WER (ep0 → chọn) | Dev >20s D (ep0 → chọn) | Peak VRAM (MB) |",
                  "| --- | --- | --- | --- | ---: | --- | --- | --- | ---: |"]
        for r in sorted(diag_fts, key=lambda x: x.get("duration_policy", "")):
            res = optional_json(work / "kaggle" / r["run"] / "output/diag-ft.json")
            if not res:
                continue
            df = res["duration_filter"]
            sc0 = res["scores_by_epoch"][0]
            sc_sel = next(s for s in res["scores_by_epoch"] if s["epoch"] == res["selected_epoch"])
            b0 = sc0.get("by_duration_band", {}).get(">20s", {})
            bs = sc_sel.get("by_duration_band", {}).get(">20s", {})
            exp = res["exposure"]
            lines.append(
                f"| `{res['duration_policy']}` | `{r['run']}` | "
                f"{df['after']['count']}/{df['before']['count']} ({df['after_gt_20s']} câu >20s) | "
                f"{exp['steps']} / {exp['total_audio_hours']:.3f}h ({exp['gt_20s_cut_views']} lượt >20s) | "
                f"{res['selected_epoch']} | {pct(sc0['wer'])} → {pct(sc_sel['wer'])} | "
                f"{pct(b0.get('wer'))} → {pct(bs.get('wer'))} | "
                f"{b0.get('D', '—')} → {bs.get('D', '—')} | {res.get('peak_allocated_mb') or '—'} |"
            )
        diag_eval = optional_json(paths.get("diag_ft_evaluation", Path("__missing__")))
        if diag_eval.get("models"):
            lines += ["", "### Đánh giá xuất ONNX FP32 & INT8 cục bộ (Sherpa-ONNX Windows)", "",
                      "| Model | Tập chấm | Chế độ | WER | CS-WER | Recall Anh | >20s WER (D) | Mất phát chữ sau nửa câu (≥20s) |",
                      "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
            for mname in ("pretrained", "control_20s", "extended_45s"):
                mdata = diag_eval["models"].get(mname) or {}
                for ds in ("authorized_dev", "vi_quick", "interim_cs_quick"):
                    ddata = mdata.get(ds) or {}
                    for mode in ("offline_fp32", "offline_int8", "streaming_fp32", "streaming_int8"):
                        sc = ddata.get(mode)
                        if not sc:
                            continue
                        b20 = sc.get("by_duration_band", {}).get(">20s")
                        b20_str = f"{pct(b20['wer'])} (D={b20['D']})" if b20 else "—"
                        sl = sc.get("streaming_long")
                        sl_str = f"{sl['lost_late_emissions']}/{sl['long_rows']}" if sl and sl["long_rows"] else "—"
                        lines.append(
                            f"| `{mname}` | `{ds}` | `{mode}` | {pct(sc['wer'])} | {pct(sc.get('cs_wer'))} | "
                            f"{pct(sc.get('term_recall'))} | {b20_str} | {sl_str} |"
                        )
    lines += curation_section(optional_json(paths.get("curation_audit", Path("__missing__"))))
    lines += exploratory_section(work, research, kaggle_runs, paths)
    lock = optional_json(paths["icefall_runtime_lock"])
    if lock and any(run.get("result_manifest_sha256") == lock["result_manifest_sha256"] for run in kaggle_runs):
        lines += ["", f"Khóa offline gồm {len(lock['packages'])} phiên bản distribution từ phiên Icefall v1 "
                  "và năm wheel được kiểm SHA256. Khóa yêu cầu ảnh Kaggle có Python, CUDA, PyTorch "
                  "và các gói dựng sẵn đúng phiên bản đã ghi."]
    export_lock = optional_json(paths["icefall_export_lock"])
    if export_lock and any(run.get("result_manifest_sha256") == export_lock["result_manifest_sha256"] for run in kaggle_runs):
        lines.append(f"Khóa huấn luyện và export FLEURS ghi {len(export_lock['packages'])} distribution "
                     "và sáu wheel có SHA256, gồm ONNX Runtime 1.32.0.")
    prepared = [run for run in kaggle_runs if run.get("experiment") == "V02" and run.get("runtime_lock")
                and run.get("bundle", {}).get("members_manifest_sha256")]
    if prepared:
        run = prepared[-1]
        lines += ["", f"Gói V02 `{run['run']}` đã chuẩn bị cục bộ với LR 0,0015, seed 0 và "
                  f"{run['bundle']['train_rows']} câu. SHA256 manifest "
                  f"`{run['bundle']['train_manifest_sha256']}`."]
    lines += ["", "## Cổng chưa đạt", ""]
    for label, why in [*((None, issue) for issue in data_issues), *issues.items()]:
        if "No such file" in why or "chưa có release" in why:
            why = "Thiếu artifact hoặc release đã đăng ký"
        lines.append(f"- {label + ' · ' if label else ''}{why}")
    lines += ["- Probe bộ nhớ V03 cần các pool đã đăng ký và ba bước FP32 ở mức 200 giây/batch có câu từ 25 giây. "
              "Recipe CLI Icefall ghim hiện loại câu train trên 20 giây. "
              "Phép đo kỹ thuật FLEURS dùng trực tiếp bước huấn luyện.",
              "", "## Việc cần người dùng", "",
              "- Cung cấp văn bản quyền huấn luyện và chuyển lên Kaggle cho danh sách V02. "
              "[Phạm vi 163 bản ghi](../data/work/vietnote/v02/rights-scope.csv).",
              "- Xin hồ sơ nguồn và người nói từ đơn vị phát hành VietSpeech. VietPrism chờ tác giả phê duyệt. "
              "[VietSpeech](https://huggingface.co/datasets/NhutP/VietSpeech).",
              "- Phân công người nghe duyệt các phiếu và xác minh người nói. "
              "[Hướng dẫn duyệt và hồ sơ quyền](../docs/VIETNOTE_KAGGLE.md#phiếu-nghe-và-quyền-nguồn)."]
    lines += ["", "## Phụ thuộc theo chiến lược", "",
              "| Nhánh | Chiến lược | Tiền đề |", "| --- | --- | --- |"]
    for name, item in read_yaml(REPO / research["strategies"]).items():
        lines.append(f"| {item['branch']} | {name} | {item['prerequisites']} |")
    lines += ["", "## Hành động nghiên cứu kế tiếp", ""]
    if "parity" in issues and constants:
        lines.append("Tái tạo nhóm hằng gây sai khác từ quy trình export và kiểm parity trên cùng 113 câu, "
                     "để sửa bản re-export từ checkpoint.")
    elif "parity" in issues and isolation:
        lines.append("Kiểm chứng ảnh hưởng của từng nhóm hằng FP32 khác nhau trong encoder, "
                     "dùng các câu sai khác đã lưu trên cùng runtime.")
    elif "parity" in issues:
        lines.append("Cô lập sai khác INT8 trên subset V01 đã khóa bằng cách lượng tử hóa lại FP32 chính thức và FP32 "
                     "re-export với cùng phiên bản ONNX Runtime, rồi đối chiếu từng bộ phận encoder, decoder và joiner.")
    elif not v02 and icefall_completed:
        lines.append("Sau khi có quyền cho đúng 163 bản ghi, tái lập arm đối chứng LR 0,0045 trên Kaggle T4 "
                     "với seed 0, cùng môi trường và 489 bước. Giữ nguyên quy tắc chấm đã đăng ký.")
    elif not v02 and any(run.get("experiment") == "synthetic" and run.get("state") == "completed"
                         for run in kaggle_runs):
        lines.append("Giữ 163 bản ghi V02 ở máy cho đến khi xác minh quyền huấn luyện và chuyển lên Kaggle. "
                     "Ghim các gói phụ thuộc cho toàn bộ recipe Icefall trước khi chạy V02 với LR 0,0015.")
    elif not v02 and any(run.get("experiment") == "smoke" and run.get("state") == "completed"
                         for run in kaggle_runs):
        lines.append("Đối chiếu văn bản cho phép huấn luyện và chuyển đúng 163 bản ghi V02 lên Kaggle, "
                     "rồi kiểm tra bộ phụ thuộc trên môi trường GPU đã đo trước khi chạy LR 0,0015.")
    elif not v02:
        lines.append("Xác minh GPU bằng Kaggle smoke sau khi xác thực. Lượt khoa học tiếp theo là V02 với LR 0,0015, "
                     "giữ nguyên 163 câu, mức tiếp xúc dữ liệu và quy tắc chấm đã đăng ký.")
    elif data_issues:
        lines.append("Xác minh quyền sử dụng và người nói cho một lô hội thoại phi y khoa trước khi phân train/dev.")
    else:
        lines.append("Kiểm V03 bundle theo seed 0 và đối chiếu giờ thực tế sau lọc.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("report")
    status.add_argument("--out", type=Path)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--seed", type=int, required=True)
    bundle.add_argument("--staged", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    research = registry()
    config = X.load(REPO / research["experiment_config"])
    if args.command == "report":
        contents = report(config, research)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(contents, encoding="utf-8", newline="\n")
        else:
            print(contents, end="")
    else:
        if args.seed not in config["seeds"]:
            parser.error("Use a registered seed")
        try:
            build_bundle(config, args.seed, args.staged, args.out)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
