import re

from tracking import storage


def test_generate_run_id_includes_ms_and_sequence(monkeypatch):
    monkeypatch.setattr(storage, "_git_short_hash", lambda: "abc1234")
    monkeypatch.setattr(storage, "_git_is_dirty", lambda: False)

    run_id = storage.generate_run_id()

    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}_abc1234_s\d{4}$",
        run_id,
    )


def test_generate_run_id_stress_uniqueness(monkeypatch):
    monkeypatch.setattr(storage, "_git_short_hash", lambda: "abc1234")
    monkeypatch.setattr(storage, "_git_is_dirty", lambda: False)

    num_ids = 10000
    ids = {storage.generate_run_id() for _ in range(num_ids)}

    assert len(ids) == num_ids
