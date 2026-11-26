"""Run Icefall's CLI with batch exposure logging, explicit duration filtering, and optional RNG state."""

import argparse
import json
import random
import sys
from pathlib import Path

UPSTREAM_MIN_DURATION = 1.0
UPSTREAM_MAX_DURATION = 20.0


class LoggedBatches:
    def __init__(self, loader, log, params):
        self.loader = loader
        self.sampler = loader.sampler
        self.log = log
        self.params = params

    def __iter__(self):
        for batch in self.loader:
            cuts = batch["supervisions"]["cut"]
            rows = [{"id": cut.id, "recording_id": cut.recording_id, "seconds": cut.duration}
                    for cut in cuts]
            yield batch
            self.log.write(json.dumps(dict(epoch=self.params.cur_epoch, step=self.params.batch_idx_train,
                                           cuts=rows)) + "\n")
            self.log.flush()


def _duration_summary(durations):
    if not durations:
        return {"count": 0, "hours": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(durations)
    n = len(ordered)

    def pct(p):
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return round(ordered[idx], 4)

    return {
        "count": n,
        "hours": round(sum(ordered) / 3600.0, 6),
        "min": round(ordered[0], 4),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": round(ordered[-1], 4),
    }


def _install_duration_policy(finetune, min_dur, max_dur, exp_dir, bpe_model):
    import sentencepiece as spm
    from lhotse import CutSet

    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_model))
    orig_train_cuts = finetune.LibriSpeechAsrDataModule.train_cuts

    def wrapped_train_cuts(self):
        raw_cuts = list(orig_train_cuts(self))
        raw_durs = [float(c.duration) for c in raw_cuts]

        class _PolicyCutSet:
            def __init__(self, items):
                self._items = items

            def __str__(self):
                return f"CutSet(len={len(self._items)})"

            def __len__(self):
                return len(self._items)

            def __iter__(self):
                return iter(self._items)

            def filter(self, predicate):
                # Replace upstream hardcoded 1.0..20.0 filter with explicit policy and record stats.
                kept, kept_durs = [], []
                dropped_short = dropped_long = dropped_t_lt_s = 0
                for c in self._items:
                    d = float(c.duration)
                    if d < min_dur:
                        dropped_short += 1
                        continue
                    if d > max_dur:
                        dropped_long += 1
                        continue
                    frames_sub = ((int(d * 100) - 7) // 2 + 1) // 2
                    tokens = sp.encode(c.supervisions[0].text, out_type=str)
                    if frames_sub < len(tokens):
                        dropped_t_lt_s += 1
                        continue
                    kept.append(c)
                    kept_durs.append(d)
                stats = {
                    "policy": {
                        "min_utt_duration": min_dur,
                        "max_utt_duration": max_dur,
                        "is_upstream_control": (min_dur == UPSTREAM_MIN_DURATION and max_dur == UPSTREAM_MAX_DURATION),
                    },
                    "before": _duration_summary(raw_durs),
                    "after": _duration_summary(kept_durs),
                    "dropped": {
                        "below_min": dropped_short,
                        "above_max": dropped_long,
                        "frames_lt_tokens": dropped_t_lt_s,
                    },
                    "before_gt_20s": sum(1 for d in raw_durs if d > 20.0),
                    "after_gt_20s": sum(1 for d in kept_durs if d > 20.0),
                }
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                (Path(exp_dir) / "duration-filter.json").write_text(
                    json.dumps(stats, indent=2) + "\n", encoding="utf-8"
                )
                return CutSet.from_cuts(kept)

        return _PolicyCutSet(raw_cuts)

    finetune.LibriSpeechAsrDataModule.train_cuts = wrapped_train_cuts


def _install_rng_resume(finetune):
    import numpy as np
    import torch

    orig_save = finetune.save_checkpoint
    orig_load = finetune.load_checkpoint_if_available
    orig_scan = finetune.scan_pessimistic_batches_for_oom

    def _rng_path(exp_dir, tag):
        return Path(exp_dir) / f"rng-{tag}.pt"

    def save_with_rng(*args, **kwargs):
        orig_save(*args, **kwargs)
        params = kwargs.get("params") if "params" in kwargs else args[0]
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, _rng_path(params.exp_dir, f"epoch-{params.cur_epoch}"))

    def load_with_rng(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        params = kwargs.get("params") if "params" in kwargs else args[0]
        tag = f"checkpoint-{params.start_batch}" if params.start_batch > 0 else f"epoch-{params.start_epoch - 1}"
        path = _rng_path(params.exp_dir, tag)
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
            if state.get("torch_cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        return result

    def safe_scan(*args, **kwargs):
        params = kwargs.get("params") if "params" in kwargs else args[4]
        if params.start_batch > 0:
            # Avoid consuming the restored mid-epoch sampler iterator during OOM scan.
            return
        return orig_scan(*args, **kwargs)

    finetune.save_checkpoint = save_with_rng
    finetune.load_checkpoint_if_available = load_with_rng
    finetune.scan_pessimistic_batches_for_oom = safe_scan


def main():
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--min-utt-duration", type=float, default=UPSTREAM_MIN_DURATION)
    wrapper.add_argument("--max-utt-duration", type=float, default=UPSTREAM_MAX_DURATION)
    wrapper.add_argument("--save-rng-state", action="store_true")
    wargs, rest = wrapper.parse_known_args(sys.argv[1:])
    if not rest:
        raise ValueError("Expected Icefall recipe script path")
    recipe_script = Path(rest[0]).resolve()
    sys.argv = [str(recipe_script), *rest[1:]]
    sys.path.insert(0, str(recipe_script.parent))
    import finetune

    if Path(finetune.__file__).resolve() != recipe_script:
        raise ValueError("Unexpected Icefall training module")

    # Peek at --exp-dir and --bpe-model for duration logging.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--exp-dir", type=Path, default=Path("zipformer/exp"))
    peek.add_argument("--bpe-model", type=Path, required=True)
    pargs, _ = peek.parse_known_args(sys.argv[1:])
    _install_duration_policy(finetune, wargs.min_utt_duration, wargs.max_utt_duration, pargs.exp_dir, pargs.bpe_model)
    if wargs.save_rng_state:
        _install_rng_resume(finetune)

    epoch_loop = finetune.train_one_epoch

    def logged_epoch(**kwargs):
        params = kwargs["params"]
        # Open a separate attempt log so a resume preserves earlier exposure records.
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode="w", dir=params.exp_dir, prefix=f"exposure-epoch-{params.cur_epoch}-",
                                suffix=".jsonl", delete=False, encoding="utf-8") as log:
            kwargs["train_dl"] = LoggedBatches(kwargs["train_dl"], log, params)
            return epoch_loop(**kwargs)

    finetune.train_one_epoch = logged_epoch
    finetune.main()


if __name__ == "__main__":
    main()
