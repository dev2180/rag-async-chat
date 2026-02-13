from pathlib import Path
from app.ingestion.hasher import compute_file_hash


def test_hash_consistency(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")

    hash1 = compute_file_hash(file_path)
    hash2 = compute_file_hash(file_path)

    assert hash1 == hash2


def test_hash_changes_on_modification(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    hash1 = compute_file_hash(file_path)

    file_path.write_text("hello world")

    hash2 = compute_file_hash(file_path)

    assert hash1 != hash2
