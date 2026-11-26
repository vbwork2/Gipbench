"""Measure the FP32 VietNote training step on released FLEURS long batches."""


def run_fleurs_memory(config, root, out, recipe, icefall, initial):
    import torch

    inputs = root / "inputs"
    extract_inputs(config, inputs)
    sys.path[:0] = [str(recipe), str(icefall), str(inputs / "src"), str(inputs / "normalizer/src"),
                    str(inputs / "scripts/diagnostics")]
    from gipbench.core.manifest import read_jsonl
    from vietnote_train_validation import measure_memory_case

    base = inputs / "models/vietnote78"
    (base / "pytorch").mkdir(parents=True)
    shutil.move(initial, base / "pytorch/pretrained.pt")
    shutil.move(root / "bpe.model", base / "bpe.model")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((inputs / "batches.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in read_jsonl(inputs / "train.jsonl")}
    groups = [[rows[key] for key in group] for group in plan["groups"]]
    limit = plan["limit"]
    if len(groups) != config["bundle"]["planned_steps"] or any(
        len(group) != len({row["id"] for row in group})
        or max(row["duration"] for row in group) < plan["long_clip_min_seconds"]
        or not limit * plan["min_batch_fraction"] <= len(group) * max(row["duration"] for row in group) <= limit
        for group in groups
    ):
        raise ValueError("The planned batches do not satisfy the FP32 long-batch probe")
    for name in ("batches.json", "train.jsonl"):
        shutil.copyfile(inputs / name, out / name)
    settings = {"tiny": {"lr": plan["base_lr"], "seed": plan["seed"]}}
    result = measure_memory_case(base, settings, "long", limit, groups, plan["min_free_fraction"])
    result.update(scope="engineering_only", v03_gate_passed=False, planned_steps=len(groups),
                  release=plan["release"], release_sha256=plan["release_sha256"], seed=plan["seed"],
                  base_lr=plan["base_lr"], gpu=torch.cuda.get_device_name(0),
                  batch_plan_sha256=digest(out / "batches.json"))
    write_json(out / "engineering-memory.json", result)
    print(json.dumps(result), flush=True)
