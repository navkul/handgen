from pathlib import Path

import pytest

from handgen.utils import sha_file
from handgen.vatr.runner import VATrRunner

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("safetensors")


def test_adapter_checks_writer_and_weights_before_load(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setenv("HANDGEN_VATR_CHECKPOINT", str(checkpoint))
    adapter = tmp_path / "writer.safetensors"
    metadata = {"format": "handgen_vatr_memory_v1", "writer_id": "writer_a", "checkpoint_sha256": sha_file(checkpoint)}
    save_file({"memory": torch.zeros(4, 1, 512)}, adapter, metadata=metadata)
    with pytest.raises(ValueError, match="different writer or checkpoint"):
        VATrRunner([], 11, "writer_b", adapter)
    checkpoint.write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="different writer or checkpoint"):
        VATrRunner([], 11, "writer_a", adapter)


def test_external_score_excludes_seen_words_and_rejects_reused_image(tmp_path):
    from handgen.vatr.fit import _external_test

    root = Path(__file__).resolve().parents[1]
    image = root / "data/style_refs/writer_001/natural_prose_alt03.png"
    transcript = root / "examples/transcripts/writer_001_holdout.txt"

    class TestGenerator:
        device = "cpu"

        def pixels(self, memory, word):
            result = torch.ones(32, len(word) * 16)
            result[11:17, 3:-3] = 0
            return result

    enrolled = {"Our": [{"source_sha256": "another-image"}]}
    score = _external_test(TestGenerator(), None, None, (image, transcript), enrolled, tmp_path)
    assert [record["word"] for record in score["words"]] == ["business", "will", "should", "Objectives"]
    assert score["baseline_loss"] == score["selected_loss"]
    assert (tmp_path / "test_comparison.png").is_file()
    enrolled["Our"][0]["source_sha256"] = sha_file(image)
    with pytest.raises(ValueError, match="separate handwriting image"):
        _external_test(TestGenerator(), None, None, (image, transcript), enrolled, tmp_path)
