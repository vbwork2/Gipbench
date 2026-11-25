"""Report VietNote research status or assemble the verified V03 training inputs."""

import argparse
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.manifest import read_jsonl
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
    return lines


def report(config, research):
    paths = {key: REPO / path for key, path in research["evidence"].items()}
    corpus, data_issues = data_ready(config, research)
    try:
        detail, issues = model_validation(config, research)
    except (OSError, ValueError) as exc:
        detail, issues = {}, {"checkpoint": str(exc)}
    quick = quick_results(paths["quick_baseline"])
    medical = optional_json(paths["medical_candidates"])
    diagnostic = optional_json(paths["diagnostic_dev"])
    historical = optional_json(paths["parity_before_softmax"])
    isolation = int8_isolation(paths, historical)
    historical_official = historical.get("official_files", {})
    constants = encoder_constants(paths, isolation, historical_official)
    v01 = all(key in detail and key not in issues for key in ("parity", "tensors")) and "export" not in issues
    v02 = all(key in detail and key not in issues for key in ("tiny", "memory"))
    statuses = {
        "V00": f"Tích hợp và quick baseline có {len(quick)} lượt đã lưu. Full dev còn mở." if quick else "Thiếu quick baseline.",
        "V01": "Đạt." if v01 else "Một phần. Xem đối chiếu FP32, INT8 và cổng bên dưới.",
        "V02": "Đạt." if v02 else "Một phần. Xem tiny-set và batch FP32 bên dưới.",
        "V03": "Bị chặn bởi cổng dữ liệu hoặc model." if data_issues or not v01 or not v02 else
               "Các tiền đề dữ liệu và model đạt. Chờ bundle và kết quả.",
    }
    lines = ["# Trạng thái nghiên cứu VietNote", "",
             "Mục tiêu giảm lỗi chuyển mã Việt–Anh với thuật ngữ chưa thấy và giữ độ chính xác tiếng Việt.", "",
             "| Lượt | Trạng thái | Điều kiện trước |", "| --- | --- | --- |"]
    for name, item in research["experiments"].items():
        lines.append(f"| {name} | {statuses.get(name, 'Chờ kết quả V03 và điều kiện trước.')} "
                     f"| {', '.join(item['requires']) or '—'} |")
    lines += ["", "## Dữ liệu V03", "", "| Nhóm | Giờ có sẵn | Giờ đủ bằng chứng | Mục tiêu | Thiếu trước lọc |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for kind, target in config["research"]["target_unique_hours"].items():
        available, eligible = corpus["available"][kind], corpus["eligible"][kind]
        lines.append(f"| {kind.upper()} | {available:.3f} | {eligible:.3f} | {target:g} | {max(0, target - available):.3f} |")
    lines += ["", "Giờ có sẵn dùng audio tại chỗ, khử trùng theo SHA256 ghi trong manifest hoặc đường dẫn. "
              "Giờ đủ bằng chứng cần hồ sơ quyền sử dụng, kiểm nhãn, phân tách người nói và kiểm âm học của release.", "",
              "| Release trong hỗn hợp | Số dòng | Giờ có sẵn | Giờ đủ bằng chứng |",
              "| --- | ---: | ---: | ---: |"]
    for ref, item in corpus["releases"].items():
        lines.append(f"| {ref} | {item['rows']} | {item['available_hours']:.3f} | {item['eligible_hours']:.3f} |")
    lines += ["", "### Tiến độ lấy dữ liệu", "", "| Nguồn | Trạng thái registry | Giờ cục bộ | Giờ đủ V03 |",
              "| --- | --- | ---: | ---: |"]
    sources = read_yaml(REPO / research["sources"])["sources"]
    work = paths["single_step"].parents[1]
    for name, item in sources.items():
        acquisition = optional_json(work / "pilots" / name / "acquisition.json")
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
        lines.append(f"| {name} | {item['status']} | {local} | {eligible:.3f} |")
    source_review = optional_json(paths["source_review"])
    if source_review.get("source_registry_sha256") == R.sha256(REPO / research["sources"]):
        queues = source_review["local_queue"]
        lines += ["", f"Hàng đợi kiểm nguồn có {queues['vss']['queued_rows']} câu VietSuperSpeech "
                  f"({queues['vss']['queued_hours']:.3f} giờ) và {queues['lsvsc']['queued_rows']} câu LSVSC "
                  f"({queues['lsvsc']['queued_hours']:.3f} giờ). Chưa bổ sung giờ đủ điều kiện V03."]
        if source_review["cards"].get("vietspeech", {}).get("gated"):
            lines.append("Metadata ở revision đã khóa xác nhận VietSpeech yêu cầu mở quyền trên Hugging Face.")
    lines += ["", "### Dev CS độc lập", "",
              f"Y khoa còn {medical.get('filter', {}).get('n_keep', '—')} ứng viên qua lọc ban đầu, "
              f"{medical.get('speakers', '—')} người nói xác minh theo báo cáo ứng viên. "
              f"Release độc lập cần {research['independent_cs']['min_rows']} câu, "
              f"{research['independent_cs']['min_domains']} domain và "
              f"{research['independent_cs']['min_verified_speakers']} người nói đã xác minh.",
              f"Quick CS ({diagnostic.get('rows', '—')} câu) dùng chẩn đoán kỹ thuật. "
              "Cấm dùng cho kết luận tổng quát hóa hoặc chấp nhận V03.", "", "## Kết quả", ""]
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
    lines += ["", "## Cổng chưa đạt", ""]
    for label, why in [*((None, issue) for issue in data_issues), *issues.items()]:
        if "No such file" in why or "chưa có release" in why:
            why = "Thiếu artifact hoặc release đã đăng ký"
        lines.append(f"- {label + ' · ' if label else ''}{why}")
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
    elif not v02:
        lines.append("Tiếp tục lượt V02 giảm learning rate đã đăng ký khi WSL truy cập lại được GPU, "
                     "giữ nguyên 163 câu và quy tắc chấm.")
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
            args.out.write_text(contents, encoding="utf-8")
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
