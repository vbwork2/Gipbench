"""Run authorized Kaggle T4 diagnostics: tiny-diag (LR A/B) and diag-ft (duration policy + E2E)."""


def _duration_band(seconds):
    if seconds < 10.0:
        return "<10s"
    if seconds <= 20.0:
        return "10-20s"
    return ">20s"


def _score_by_band(rows, hyps, catalog, profile):
    from gipbench.bench.evaluate import score_rows, summarize

    recs = score_rows(rows, hyps, catalog, profile, "text_tagged", "speaker", [])
    overall = summarize(recs, 0, 0, with_ci=False)
    overall.update({k: sum(r[k] for r in recs) for k in ("S", "D", "I")})
    row_by_id = {r["id"]: r for r in rows}
    bands = {}
    for band in ("<10s", "10-20s", ">20s"):
        sub_recs = [r for r in recs if _duration_band(float(row_by_id[r["id"]]["duration"])) == band]
        if sub_recs:
            s = summarize(sub_recs, 0, 0, with_ci=False)
            s.update({k: sum(r[k] for r in sub_recs) for k in ("S", "D", "I")})
            bands[band] = s
    overall["by_duration_band"] = bands
    return overall


def run_tiny_diag(config, root, out, recipe, icefall, initial):
    import torch

    inputs = root / "inputs"
    extract_inputs(config, inputs)
    sys.path[:0] = [str(recipe), str(icefall), str(inputs / "src"), str(inputs / "normalizer/src"),
                    str(inputs / "scripts/diagnostics")]
    from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
    from vietnote_train_validation import tiny

    base = inputs / "models/vietnote78"
    (base / "pytorch").mkdir(parents=True)
    shutil.move(initial, base / "pytorch/pretrained.pt")
    shutil.move(root / "bpe.model", base / "bpe.model")
    train = read_jsonl(inputs / "train.jsonl")
    digest_train = manifest_sha256(train)
    if digest_train != config["bundle"]["train_manifest_sha256"]:
        raise ValueError("Authorized tiny-diag manifest identity mismatch")
    settings = {
        "rows": len(train),
        "seed": 0,
        "passes": 3,
        "batch_seconds": 16,
        "lr": float(config["lr"]),
        "relative_wer_reduction_min": 0.20,
        "relative_cs_wer_reduction_min": 0.20,
        "recall_drop_max_pp": 0,
    }
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_jsonl(out / "train.jsonl", train)
    result = tiny(base, train, settings, out)
    # Strict reload and checkpoint digest are already verified inside tiny(); drop large weights for compact sync.
    (out / "tiny.pt").unlink(missing_ok=True)
    result.update(
        experiment="tiny-diag",
        not_v02=True,
        arm=config["arm"],
        lr=float(config["lr"]),
        seed=0,
        precision="fp32",
        train_rows=len(train),
        train_hours=sum(float(row["duration"]) for row in train) / 3600.0,
        train_manifest_sha256=digest_train,
        license=config["bundle"]["license"],
        gpu=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),
    )
    write_json(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("arm", "lr", "steps", "passed", "training_summary", "score_by_pass")}),
          flush=True)


def _run_icefall_ft_and_eval(config, root, out, recipe, icefall, initial, env, *, min_dur, max_dur,
                               max_duration, num_buckets, num_epochs, base_lr, experiment_name, scope,
                               report_filename, extra_fields=None):
    import random
    import re
    import torch
    from lhotse import CutSet, Recording, SupervisionSegment

    inputs = root / "inputs"
    extract_inputs(config, inputs)
    sys.path[:0] = [str(recipe), str(icefall), str(inputs / "src"), str(inputs / "normalizer/src"),
                    str(inputs / "scripts/diagnostics")]
    from beam_search import greedy_search_batch
    from gipbench.bench.evaluate import scorer_code_sha256
    from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
    from gipbench.core.normprofiles import get_profile
    from gipbench.core.textnorm import Catalog, plain_text
    from vnasr.text import normalize_text
    from vietnote_torch_check import load_model
    from vietnote_train_validation import batches, make_batch, prepare

    base = inputs / "models/vietnote78"
    (base / "pytorch").mkdir(parents=True)
    shutil.copyfile(initial, base / "pytorch/pretrained.pt")
    shutil.copyfile(root / "bpe.model", base / "bpe.model")

    train_rows = read_jsonl(inputs / "train.jsonl")
    valid_rows = read_jsonl(inputs / "valid.jsonl")
    if (manifest_sha256(train_rows) != config["bundle"]["train_manifest_sha256"]
            or manifest_sha256(valid_rows) != config["bundle"]["valid_manifest_sha256"]):
        raise ValueError(f"{experiment_name} manifest identity mismatch")

    cuts_dir = root / "cuts"
    cuts_dir.mkdir()
    for split_name, rows in (("train", train_rows), ("valid", valid_rows)):
        cuts = []
        for row in rows:
            audio = Path(row["_audio_path"]).resolve()
            if digest(audio) != row["audio_sha256"]:
                raise ValueError(f"Audio digest mismatch in {split_name}: {row['id']}")
            rec = Recording.from_file(audio, recording_id=row["id"])
            cut = rec.to_cut()
            cut.supervisions = [SupervisionSegment(
                id=row["id"], recording_id=rec.id, start=0.0, duration=cut.duration,
                text=normalize_text(plain_text(row["text_tagged"])),
            )]
            cuts.append(cut)
        CutSet.from_cuts(cuts).to_file(cuts_dir / f"{split_name}.jsonl.gz")
        write_jsonl(out / f"{split_name}.jsonl", rows)

    exp_dir = root / "exp"
    exp_dir.mkdir()
    cmd = [
        sys.executable, str(inputs / "scripts/diagnostics/vietnote_logged_cli.py"),
        "--min-utt-duration", str(min_dur),
        "--max-utt-duration", str(max_dur),
        str(recipe / "finetune.py"),
        "--world-size", "1", "--seed", "0",
        "--exp-dir", str(exp_dir), "--manifest-dir", str(cuts_dir),
        "--bpe-model", str(root / "bpe.model"),
        "--num-epochs", str(num_epochs), "--start-epoch", "1",
        "--do-finetune", "1", "--finetune-ckpt", str(initial),
        "--num-encoder-layers", "2,2,3,4,3,2", "--encoder-dim", "192,256,384,512,384,256",
        "--feedforward-dim", "512,768,1024,1536,1024,768", "--decoder-dim", "512", "--joiner-dim", "512",
        "--causal", "1", "--chunk-size", "16,32,64,-1", "--left-context-frames", "64,128,256,-1",
        "--use-fp16", "0", "--use-mux", "0",
        "--bucketing-sampler", "1", "--num-buckets", str(num_buckets), "--drop-last", "0", "--num-workers", "2",
        "--max-duration", str(max_duration), "--on-the-fly-feats", "1", "--enable-musan", "0", "--tensorboard", "0",
        "--base-lr", str(base_lr), "--lr-batches", "100000", "--lr-epochs", "100",
        "--save-every-n", "5000", "--keep-last-k", "5",
    ]
    run_env = dict(env, PYTHONPATH=os.pathsep.join([str(recipe), str(icefall), str(inputs / "src"),
                                                    str(inputs / "normalizer/src")]))
    train_log = out / "train.log"
    with train_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=root, env=run_env, stdout=handle, stderr=subprocess.STDOUT)
    log_text = train_log.read_text(encoding="utf-8", errors="replace")
    if proc.returncode:
        raise RuntimeError(log_text[-5000:])

    dur_filter_file = exp_dir / "duration-filter.json"
    if not dur_filter_file.is_file():
        raise RuntimeError("Missing duration-filter.json from logged CLI")
    duration_filter = json.loads(dur_filter_file.read_text(encoding="utf-8"))
    shutil.copyfile(dur_filter_file, out / "duration-filter.json")

    exposure_steps = 0
    exposure_seconds = 0.0
    exposure_gt20_cut_views = 0
    exposure_unique_ids = set()
    for exp_file in sorted(exp_dir.glob("exposure-epoch-*.jsonl")):
        shutil.copyfile(exp_file, out / exp_file.name)
        for line in exp_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            exposure_steps += 1
            for cut in rec["cuts"]:
                sec = float(cut["seconds"])
                exposure_seconds += sec
                exposure_unique_ids.add(cut["recording_id"])
                if sec > 20.0:
                    exposure_gt20_cut_views += 1

    valid_losses = [float(v) for v in re.findall(r"validation: loss=([0-9.eE+-]+)", log_text)]
    peak_mb_matches = [int(v) for v in re.findall(r"Maximum memory allocated so far is (\d+)MB", log_text)]

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev_features = prepare(valid_rows)
    catalog_path = inputs / "data/releases/catalog/train-v1/catalog.tsv"
    catalog = Catalog.load(str(catalog_path))
    profile = get_profile("gipbench_strict_v1")
    model, sp, _ = load_model(base, offline=True)
    model = model.cuda().eval()

    scores = []
    for epoch in range(num_epochs + 1):
        ckpt_path = initial if epoch == 0 else exp_dir / f"epoch-{epoch}.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        hyps = {}
        with torch.inference_mode():
            for batch in batches(valid_rows, 48, random.Random(0)):
                tensors = make_batch(batch, dev_features)
                enc, lengths = model.forward_encoder(tensors["inputs"].cuda(), tensors["supervisions"]["num_frames"].cuda())
                for row, tokens in zip(batch, greedy_search_batch(model, enc, lengths), strict=True):
                    hyps[row["id"]] = sp.decode(tokens)
        hyps_path = out / f"epoch-{epoch}-dev.jsonl"
        write_jsonl(hyps_path, ({"id": row["id"], "hyp": hyps[row["id"]]} for row in valid_rows))
        metrics = _score_by_band(valid_rows, hyps, catalog, profile)
        scores.append({
            "epoch": epoch,
            "steps": int(state.get("batch_idx_train", 0)),
            "hyps_sha256": digest(hyps_path),
            **metrics,
        })
        del state

    selected = min(
        scores[1:],
        key=lambda s: (s["wer"], s["by_duration_band"].get(">20s", {}).get("D", 0), s["epoch"]),
    )
    sel_ckpt = torch.load(exp_dir / f"epoch-{selected['epoch']}.pt", map_location="cpu", weights_only=False)
    selected_out = out / "selected.pt"
    torch.save({"model": sel_ckpt["model"]}, selected_out)
    del sel_ckpt

    report = {
        "experiment": experiment_name,
        "scope": scope,
        "min_utt_duration": min_dur,
        "max_utt_duration": max_dur,
        "seed": 0,
        "base_lr": base_lr,
        "max_duration": max_duration,
        "num_epochs": num_epochs,
        "precision": "fp32",
        "license": config["bundle"]["license"],
        "train_rows_input": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_manifest_sha256": config["bundle"]["train_manifest_sha256"],
        "valid_manifest_sha256": config["bundle"]["valid_manifest_sha256"],
        "duration_filter": duration_filter,
        "exposure": {
            "steps": exposure_steps,
            "total_audio_seconds": round(exposure_seconds, 3),
            "total_audio_hours": round(exposure_seconds / 3600.0, 6),
            "unique_recordings_seen": len(exposure_unique_ids),
            "gt_20s_cut_views": exposure_gt20_cut_views,
        },
        "validation_losses": valid_losses,
        "peak_allocated_mb": max(peak_mb_matches) if peak_mb_matches else None,
        "scores_by_epoch": scores,
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_sha256": digest(selected_out),
        "scorer_code_sha256": scorer_code_sha256(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        **(extra_fields or {}),
    }
    write_json(out / report_filename, report)
    print(json.dumps({
        "experiment": experiment_name,
        "selected_epoch": selected["epoch"],
        "exposure": report["exposure"],
        "epoch0_wer": scores[0]["wer"],
        "selected_wer": selected["wer"],
        "epoch0_gt20_wer": scores[0]["by_duration_band"].get(">20s", {}).get("wer"),
        "selected_gt20_wer": selected["by_duration_band"].get(">20s", {}).get("wer"),
    }), flush=True)


def run_diag_ft(config, root, out, recipe, icefall, initial, env):
    policy = config["duration_policy"]
    min_dur, max_dur = (1.0, 20.0) if policy == "control_20s" else (1.0, 45.0)
    _run_icefall_ft_and_eval(
        config, root, out, recipe, icefall, initial, env,
        min_dur=min_dur, max_dur=max_dur, max_duration=120, num_buckets=5, num_epochs=3, base_lr=0.0015,
        experiment_name="diag-ft", scope="authorized_diagnostic_not_v03", report_filename="diag-ft.json",
        extra_fields={"duration_policy": policy},
    )


def run_expl_ft(config, root, out, recipe, icefall, initial, env):
    _run_icefall_ft_and_eval(
        config, root, out, recipe, icefall, initial, env,
        min_dur=1.0, max_dur=35.0, max_duration=160, num_buckets=8, num_epochs=3, base_lr=0.0015,
        experiment_name="expl-ft", scope="exploratory_not_v03", report_filename="expl-ft.json",
        extra_fields={"source_hours": config["bundle"].get("source_hours", {})},
    )
