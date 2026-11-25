"""Cover the V03 gates that the historical model runner does not exercise."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gipbench.core import release as R
from gipbench.model import readiness as V
from gipbench.model.families import family


class ReadinessTests(unittest.TestCase):
    def test_model_family_contracts(self):
        self.assertEqual(family({})["pipeline_stages"], list(range(7)))
        self.assertEqual(family({"family": "vietnote_zipformer"})["pipeline_stages"], list(range(4)))
        with self.assertRaisesRegex(ValueError, "Unknown model family"):
            family({"family": "typo"})

    def test_approved_release_requires_pinned_rights_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = Path(tmp) / "rights.txt"
            proof.write_text("Training permission for the pinned recording source", encoding="utf-8")
            sources = {"fixture": {"revision": "r1", "recording_rights": {
                "evidence": str(proof), "sha256": R.sha256(proof), "license": "CC-BY-4.0", "training_allowed": True}}}
            release = R.Release("pool/example@1", proof.parent, {"flags": {
                "audio_screened": True, "acoustic_checked": True, "vietnote_v03": {
                    "source": "fixture", "source_revision": "r1", "kind": "vi",
                    "transcripts_verified": True, "speakers_verified": True}}})
            self.assertFalse(V.eligible_release(release, sources, "vi"))
            row = dict(id="one", source="source", speaker="speaker", domain="news", speech_type="natural",
                       language_category="vi", split="train", audio_sha256="audio", text_tagged="xin chào")
            (release.dir / "manifest.jsonl").write_text(json.dumps(row) + "\n")
            review = release.dir / "review.json"
            review.write_text(json.dumps(dict(release=release.ref, source_revision="r1", reviewer="listener",
                              manifest_sha256=R.sha256(release.manifest), decisions={
                                  "one": dict(row, language="vi", decision="accept")})))
            sources["fixture"]["release_reviews"] = {release.ref: {"path": str(review), "sha256": R.sha256(review)}}
            self.assertTrue(V.eligible_release(release, sources, "vi"))
            self.assertFalse(V.eligible_release(release, sources, "cs"))
            proof.write_text("Changed", encoding="utf-8")
            self.assertFalse(V.eligible_release(release, sources, "vi"))

    def test_memory_checks_only_selected_workload(self):
        spec = dict(precision="fp32", steps_per_case=1, min_batch_fraction=.8, min_free_fraction=.1,
                    long_clip_min_seconds=25)
        case = dict(limit=32, precision="fp32", passed=True, peak_allocated_bytes=50, peak_reserved_bytes=60,
                    gpu_total_bytes=100, gpu_free_before_bytes=90, longest_clip=30,
                    steps=[dict(audio_seconds=30, padded_seconds=30)])
        evidence = dict(selected_limit=32, gpu="local", environment={"colab": False}, passed=False,
                        cases=[dict(case, kind="pooled"), dict(case, kind="long"), dict(kind="pooled", limit=200, passed=False)])
        train = dict(max_duration=32, fp16=False)
        V.check_memory(evidence, spec, train, "local")
        V.check_memory(evidence, spec, train, "local", available_vram=(70, 100))
        with self.assertRaisesRegex(ValueError, "Current CUDA memory"):
            V.check_memory(evidence, spec, train, "local", available_vram=(69, 100))
        with self.assertRaises(ValueError):
            V.check_memory(evidence, spec, train, "T4")
        with self.assertRaises(ValueError):
            V.check_memory(evidence, spec, dict(train, max_duration=200), "local")
        with self.assertRaises(ValueError):
            V.check_memory(evidence, spec, dict(train, fp16=True), "local")
        evidence["cases"][1]["peak_allocated_bytes"] = 150
        with self.assertRaises(ValueError):
            V.check_memory(evidence, spec, train, "local")

    def test_missing_dev_and_inconsistent_mixture(self):
        with self.assertRaisesRegex(ValueError, "missing 1"):
            V.require_dev_coverage({"vi1", "vi2", "cs1"}, {"vi1", "cs1"})
        groups = [dict(group="replay_real", weight=.6), dict(group="cs_real", weight=.4)]
        with patch.object(V.X, "groups", return_value=groups):
            with self.assertRaisesRegex(ValueError, "registered mixture"):
                V.check_mixture({"features": {"max_duration": 45}}, {"vi": 18, "cs": 8})

    def test_bundle_rejects_stale_release_or_cutset(self):
        config = {"research": {"target_unique_hours": {"vi": 12, "cs": 8}}}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cuts = {}
            for name in ("train.jsonl.gz", "valid.jsonl.gz"):
                path = directory / name
                path.write_bytes(b"test CutSet")
                cuts[name] = R.sha256(path)
            (directory / "leak_report.json").write_text('{"passed": true}', encoding="utf-8")
            reserved = directory / "reserved.txt"
            reserved.write_text("reserved", encoding="utf-8")
            bindings = {"config_sha256": "locked", "release_sha256": {"pool/example@1": "fixed"}}
            evidence = dict(bindings, cutsets=cuts, hours={"vi": 12, "cs": 8},
                            reserved_terms_sha256=R.sha256(reserved),
                            contamination_sha256=R.sha256(directory / "leak_report.json"))
            (directory / "bundle.json").write_text(json.dumps(evidence), encoding="utf-8")
            with (patch.object(V, "bindings", return_value=bindings) as current,
                  patch.object(V.LG, "gate_config", return_value={"reserved_terms": reserved}),
                  patch.object(V, "cutset_hours", return_value=evidence["hours"])):
                self.assertEqual(V.verify_bundle(config, 0, directory), evidence)
                current.return_value = dict(bindings, release_sha256={"pool/example@1": "changed"})
                with self.assertRaisesRegex(ValueError, "Stale bundle"):
                    V.verify_bundle(config, 0, directory)
                current.return_value = bindings
                (directory / "valid.jsonl.gz").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Stale CutSets"):
                    V.verify_bundle(config, 0, directory)


if __name__ == "__main__":
    unittest.main()
