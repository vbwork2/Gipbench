from gipbench.dataset.ingest.vietnote import recover_journal, write_review_queue
from gipbench.dataset.release.vietnote_acquired import unique_candidates


def test_resume_discards_partial_journal_record_and_preserves_review(tmp_path):
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_bytes(b'{"id":"first"}\n{"id":"part')
    recover_journal(manifest)
    assert manifest.read_bytes() == b'{"id":"first"}\n'
    review = tmp_path / "review.csv"
    first = {"id": "first", "audio": "audio/a.wav", "audio_sha256": "digest", "text_tagged": "x", "duration": 3.6}
    write_review_queue(review, [first])
    review.write_text(review.read_text(encoding="utf-8-sig").replace("3.6,x,,,", "3.6,x,,accept,"),
                      encoding="utf-8-sig")
    assert write_review_queue(review, [first, dict(first, id="second")]) == 0.001
    assert write_review_queue(review, [dict(first, text_tagged="updated")]) == 0


def test_duplicate_audio_across_derivation_and_existing_pools():
    existing = [{"audio_sha256": "old", "upstream_audio_sha256": "shared"}]
    candidates = [{"id": "duplicate_existing", "audio_sha256": "recoded", "upstream_audio_sha256": "shared"},
                  {"id": "unique", "audio_sha256": "new", "upstream_audio_sha256": "new_upstream"},
                  {"id": "duplicate_candidate", "audio_sha256": "other", "upstream_audio_sha256": "new_upstream"}]
    keep, drop = unique_candidates(candidates, existing)
    assert [row["id"] for row in keep] == ["unique"]
    assert [row["id"] for row in drop] == ["duplicate_existing", "duplicate_candidate"]
