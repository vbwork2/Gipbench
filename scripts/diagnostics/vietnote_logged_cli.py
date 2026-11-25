"""Run Icefall's CLI and record the batches consumed by its epoch loop."""

import json
import sys
from pathlib import Path


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


def main():
    recipe_script = Path(sys.argv.pop(1)).resolve()
    sys.path.insert(0, str(recipe_script.parent))
    import finetune

    if Path(finetune.__file__).resolve() != recipe_script:
        raise ValueError("Unexpected Icefall training module")
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
