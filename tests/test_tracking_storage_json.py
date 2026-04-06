from tracking.storage import _read_json_array


def test_read_json_array_returns_empty_when_file_missing(tmp_path):
    missing = tmp_path / "missing.json"
    assert _read_json_array(missing) == []


def test_read_json_array_returns_empty_for_corrupted_json(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text("[{\"run_id\": \"ok\"},", encoding="utf-8")

    assert _read_json_array(path) == []


def test_read_json_array_returns_empty_for_non_array_json(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text('{"run_id": "r1"}', encoding="utf-8")

    assert _read_json_array(path) == []
