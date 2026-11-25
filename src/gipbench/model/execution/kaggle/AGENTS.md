# VietNote research execution

Use the checked-out `feature/zip-ft` branch. VietNote-Zipformer-78M is the active family. Preserve historical GipFormer baselines and locked evaluation rules.

Read `configs/research/vietnote.yaml`, the experiment config, and `reports/vietnote_status.md`. V02 is the registered lower learning-rate arm on the same 163 recordings.

- Run `doctor` with the official Kaggle CLI. Keep credentials in the user's Kaggle profile or process environment.
- Run `plan` before `prepare`. Submit the approved run with its explicit session budget and cloud transfer evidence where applicable.
- Reconcile the recorded notebook version with `status` before retrying a submission. Use `logs`, `watch`, and `sync` against that version.
- Verify the artifact manifest before recording completion. Track acceptance separately.
- A model-only checkpoint supports scoring and export. A training resume requires the optimizer, scheduler, sampler, and RNG state supported by the recipe.
- Preserve all V03 prerequisites, including independent CS development and a memory measurement of the intended GPU configuration.

See [the execution guide](../../../../../docs/VIETNOTE_KAGGLE.md).
