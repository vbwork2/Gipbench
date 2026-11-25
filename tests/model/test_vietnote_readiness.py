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
            self.assertTrue(V.eligible_release(release, sources, "vi"))
            self.assertFalse(V.eligible_release(release, sources, "cs"))
            proof.write_text("Changed", encoding="utf-8")
            self.assertFalse(V.eligible_release(release, sources, "vi"))

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
