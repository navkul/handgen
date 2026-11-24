import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from handgen.diffbrush.runner import DiffBrushRunner, ProseVerificationError


class DiffBrushRerankTests(unittest.TestCase):
    def test_run_cache_keys_actual_inputs_and_checks_sample_bytes(self):
        class Model:
            calls = 0

            def generate_one(self, _prompt, _style, path, **kwargs):
                self.calls += 1
                path.write_bytes(b"model output")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = root / "style.png"
            writer.write_bytes(b"first reference")
            model = Model()
            runner = DiffBrushRunner()
            runner.root = root
            runner.checkpoint = root / "weights.pt"
            runner.checkpoint.write_bytes(b"weights")
            runner._device = SimpleNamespace(type="cpu")
            args = {"span_id": "same", "prompt": "Notes", "style_ref": writer, "out_dir": root,
                    "seed": 23, "writer_id": "writer", "conditioning_mode": "prose"}
            stub = SimpleNamespace(seed_everything=lambda seed: None)
            with patch.dict("sys.modules", {"handgen.diffbrush.run_single": stub}), patch.object(runner, "_ensure_inner", return_value=model):
                first = runner._run(**args)
                self.assertEqual(first, runner._run(**args))
                self.assertEqual(model.calls, 1)
                (first.parent / "sample.png").write_bytes(b"incomplete")
                self.assertEqual(first, runner._run(**args))
                self.assertEqual(model.calls, 2)
                second = runner._run(**(args | {"seed": 24}))
                writer.write_bytes(b"revised reference")
                third = runner._run(**args)
                self.assertEqual(len({first, second, third}), 3)
                self.assertEqual(model.calls, 4)

    def test_unrepresentable_prose_is_reported_before_sampling(self):
        runner = DiffBrushRunner()
        with self.assertRaisesRegex(ProseVerificationError, "no glyph"):
            runner.generate_chunk(span_id="line", text="naïve", style_ref=Path("unused"),
                                  out_dir=Path("unused"), seed=2, writer_id="writer")

    def test_generate_chunk_rejects_bad_ocr_and_accepts_verified_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            runner = DiffBrushRunner()
            runner.prose_candidates = 2

            ocr_results = [{"text": "Pick Pick extra"}, {"text": "Pick"}]

            with patch.object(runner, "validate", lambda: None), patch.object(
                runner, "_run", side_effect=lambda **kwargs: self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])
            ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
            ), patch(
                "handgen.diffbrush.runner.ocr_image", side_effect=ocr_results
            ):
                run = runner.generate_chunk(
                    span_id="span",
                    text="Pick",
                    prompt_text="Pick",
                    style_ref=root / "style.png",
                    out_dir=root,
                    seed=10,
                    writer_id="writer",
                )

            self.assertTrue(run["exactness_certified"])
            self.assertEqual(run["accepted_candidate"], "span_cand01")
            self.assertEqual(run["candidate_count"], 2)
            self.assertCountEqual(run["rejected_candidates"][0]["verification"]["reject_reasons"], ["hallucinated_tokens", "repeated_tokens"])
            self.assertTrue(Path(run["styled_crop"]["path"]).exists())

    def test_generate_chunk_accepts_ocr_prefix_after_suffix_trim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            runner = DiffBrushRunner()
            runner.prose_candidates = 1
            ocr_results = [{"text": "Prove that cqy"}, {"text": "Prove that"}]
            boxes = {
                "words": [
                    {"normalized": "prove", "left": 0, "right": 30},
                    {"normalized": "that", "left": 38, "right": 70},
                    {"normalized": "cqy", "left": 95, "right": 130},
                ]
            }

            with patch.object(runner, "validate", lambda: None), patch.object(
                runner, "_run", side_effect=lambda **kwargs: self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])
            ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
            ), patch(
                "handgen.diffbrush.runner.ocr_image", side_effect=ocr_results
            ), patch("handgen.diffbrush.runner.ocr_word_boxes", return_value=boxes), patch(
                "handgen.diffbrush.runner.trim_rgba_right", side_effect=self._fake_trim
            ):
                run = runner.generate_chunk(
                    span_id="span",
                    text="Prove that",
                    style_ref=root / "style.png",
                    out_dir=root,
                    seed=10,
                    writer_id="writer",
                )

            self.assertTrue(run["exactness_certified"])
            self.assertTrue(run["ocr_verification"]["accepted_after_suffix_trim"])
            self.assertEqual(run["ocr_verification"]["trimmed_suffix_tokens"], ["cqy"])

    def test_generate_chunk_marks_low_score_best_effort_as_last_resort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            runner = DiffBrushRunner()
            runner.prose_candidates = 1
            runner.min_best_effort_score = 0.12
            runner.last_resort_retries = 0

            with patch.object(runner, "validate", lambda: None), patch.object(
                runner, "_run", side_effect=lambda **kwargs: self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])
            ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
            ), patch(
                "handgen.diffbrush.runner.ocr_image", return_value={"text": "garbled"}
            ), patch("handgen.diffbrush.runner.ocr_word_boxes", return_value={"words": []}):
                run = runner.generate_chunk(
                    span_id="span",
                    text="If value",
                    style_ref=root / "style.png",
                    out_dir=root,
                    seed=10,
                    writer_id="writer",
                )

            self.assertFalse(run["exactness_certified"])
            self.assertEqual(run["route"], "diffbrush_last_resort_prose_token")
            self.assertIn("below the configured OCR match threshold", run["best_effort_reason"])

    def test_generate_chunk_uses_style_legibility_as_best_effort_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            runner = DiffBrushRunner()
            runner.prose_candidates = 1
            runner.min_best_effort_score = 0.12
            runner.style_legibility_floor_ratio = 0.8
            runner.last_resort_retries = 0

            with patch.object(runner, "validate", lambda: None), patch.object(
                runner, "_run", side_effect=lambda **kwargs: self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])
            ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
            ), patch(
                "handgen.diffbrush.runner.ocr_legibility_score", return_value={"available": True, "score": 0.4}
            ), patch(
                "handgen.diffbrush.runner.ocr_image", return_value={"text": "Then x y z"}
            ), patch("handgen.diffbrush.runner.ocr_word_boxes", return_value={"words": []}):
                run = runner.generate_chunk(
                    span_id="span",
                    text="Then term is useful",
                    style_ref=root / "style.png",
                    out_dir=root,
                    seed=10,
                    writer_id="writer",
                )

            self.assertFalse(run["exactness_certified"])
            self.assertEqual(run["route"], "diffbrush_last_resort_prose_token")
            self.assertEqual(run["best_effort_min_score"], 0.12)
            self.assertEqual(run["best_effort_effective_min_score"], 0.32)
            self.assertGreater(run["ocr_verification"]["match_score"], run["best_effort_min_score"])
            self.assertLess(run["ocr_verification"]["match_score"], run["best_effort_effective_min_score"])

    def test_generate_chunk_reranks_across_prompt_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            runner = DiffBrushRunner()
            runner.prose_candidates = 1
            runner.last_resort_retries = 0
            ocr_results = [{"text": "garbled"}, {"text": "still garbled"}, {"text": "If number"}]

            with patch.object(runner, "validate", lambda: None), patch.object(
                runner, "_run", side_effect=lambda **kwargs: self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])
            ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
            ), patch(
                "handgen.diffbrush.runner.ocr_image", side_effect=ocr_results
            ), patch("handgen.diffbrush.runner.ocr_word_boxes", return_value={"words": []}), patch(
                "handgen.diffbrush.runner.trim_rgba_right", side_effect=self._fake_trim
            ):
                run = runner.generate_chunk(
                    span_id="span",
                    text="If number",
                    prompt_text=["If number is positive", "If number"],
                    style_ref=root / "style.png",
                    out_dir=root,
                    seed=10,
                    writer_id="writer",
                )

            self.assertTrue(run["exactness_certified"])
            self.assertEqual(run["diffbrush_prompt"], "If number")
            self.assertEqual(run["accepted_candidate"], "span_p01_cand00")
            self.assertEqual(run["candidate_count"], 2)

    def test_generate_chunk_adaptively_retries_last_resort_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            runner = DiffBrushRunner()
            runner.prose_candidates = 1
            runner.last_resort_retries = 2
            runner.min_best_effort_score = 0.12
            ocr_results = [{"text": "garbled"}, {"text": "If number and then"}, {"text": "If number"}]

            with patch.object(runner, "validate", lambda: None), patch.object(
                runner, "_run", side_effect=lambda **kwargs: self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])
            ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
            ), patch(
                "handgen.diffbrush.runner.ocr_image", side_effect=ocr_results
            ), patch("handgen.diffbrush.runner.ocr_word_boxes", return_value={"words": []}):
                run = runner.generate_chunk(
                    span_id="span",
                    text="If number",
                    style_ref=root / "style.png",
                    out_dir=root,
                    seed=10,
                    writer_id="writer",
                )

            self.assertTrue(run["exactness_certified"])
            self.assertEqual(run["accepted_candidate"], "span_cand01")
            self.assertEqual(run["candidate_count"], 2)
            self.assertTrue(run["diffbrush_prompt"].startswith("If number and then"))
            self.assertTrue(run["ocr_verification"]["estimated_padding_trim"])

    def test_prompt_seed_is_stable_across_variant_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.png"
            sample.write_bytes(b"png")
            seeds_by_prompt: dict[str, int] = {}

            def fake_result(**kwargs):
                seeds_by_prompt[kwargs["prompt"]] = kwargs["seed"]
                return self._fake_result(root, sample, kwargs["span_id"], kwargs["seed"])

            def run_with_prompts(prompts, span_id):
                runner = DiffBrushRunner()
                runner.prose_candidates = 1
                runner.min_best_effort_score = 2.0
                runner.last_resort_retries = 0
                with patch.object(runner, "validate", lambda: None), patch.object(
                    runner, "_run", side_effect=fake_result
                ), patch("handgen.diffbrush.runner.soft_diffbrush_ink", side_effect=self._fake_styled), patch(
                    "handgen.diffbrush.runner.rgba_ink_profile", return_value=self._clean_ink_profile()
                ), patch(
                    "handgen.diffbrush.runner.ocr_image", return_value={"text": "garbled"}
                ), patch("handgen.diffbrush.runner.ocr_word_boxes", return_value={"words": []}), patch(
                    "handgen.diffbrush.runner.trim_rgba_right", side_effect=self._fake_trim
                ):
                    runner.generate_chunk(
                        span_id=span_id,
                        text="If number",
                        prompt_text=prompts,
                        style_ref=root / "style.png",
                        out_dir=root,
                        seed=10,
                        writer_id="writer",
                    )

            run_with_prompts(["If number", "If number is positive"], "span_a")
            first_seed = seeds_by_prompt["If number"]
            seeds_by_prompt.clear()
            run_with_prompts(["If number is positive", "If number"], "span_b")

            self.assertEqual(seeds_by_prompt["If number"], first_seed)

    @staticmethod
    def _fake_result(root: Path, sample: Path, span_id: str, seed: int) -> Path:
        run_dir = root / "diffbrush_runs" / span_id
        run_dir.mkdir(parents=True)
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps({"outputs": {"sample": str(sample)}, "seed": seed, "steps": 1, "device": "test"}),
            encoding="utf-8",
        )
        return result_path

    @staticmethod
    def _fake_styled(_sample: Path, dst: Path, **_kwargs):
        from PIL import Image, ImageDraw

        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGBA", (140, 24))
        draw = ImageDraw.Draw(im)
        for x in [0, 38, 95]:
            draw.rectangle((x, 4, x + 29, 20), fill="black")
        im.save(dst)
        return {"path": str(dst), "width": im.width, "height": im.height, "sha256": "sha"}

    @staticmethod
    def _fake_trim(src: Path, dst: Path, *_args, **_kwargs):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return {"path": str(dst), "width": 8, "height": 10, "sha256": "trimsha"}

    @staticmethod
    def _clean_ink_profile():
        return {"ink_density": 0.18, "mean_alpha": 0.5, "ink_bbox_width": 80, "ink_bbox_height": 20}


if __name__ == "__main__":
    unittest.main()
