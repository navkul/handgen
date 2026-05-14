from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import to_pil_image


WRITER_NUMS = 496
IAM_LETTERS = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
REPO_DEFAULT = Path(__file__).resolve().parents[2] / "models" / "diffbrush" / "third_party_repo"


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def ensure_repo_importable(repo_path: Path) -> None:
    repo = str(repo_path)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def select_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized in {"", "auto"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(normalized)


def load_cfg(repo_path: Path):
    ensure_repo_importable(repo_path)
    from parse_config import assert_and_infer_cfg, cfg, cfg_from_file

    cfg.immutable(False)
    cfg_from_file(str(repo_path / "configs" / "IAM.yml"))
    assert_and_infer_cfg(make_immutable=False)
    return cfg


def build_unet(repo_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = load_cfg(repo_path)

    import models.encoder as encoder_module
    from models.unet import UNetModel

    original_resnet18 = encoder_module.models.resnet18

    def resnet18_without_preload(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original_resnet18(*args, **kwargs)

    encoder_module.models.resnet18 = resnet18_without_preload
    original_tensor_cuda = torch.Tensor.cuda

    def tensor_cuda_noop(tensor, *args, **kwargs):
        return tensor

    torch.Tensor.cuda = tensor_cuda_noop
    try:
        unet = UNetModel(
            in_channels=cfg.MODEL.IN_CHANNELS,
            model_channels=cfg.MODEL.EMB_DIM,
            out_channels=cfg.MODEL.OUT_CHANNELS,
            num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_heads=cfg.MODEL.NUM_HEADS,
            context_dim=cfg.MODEL.EMB_DIM,
            nb_classes=WRITER_NUMS,
        )
    finally:
        torch.Tensor.cuda = original_tensor_cuda
        encoder_module.models.resnet18 = original_resnet18

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    unet.load_state_dict(state_dict)
    del state_dict
    unet.to(device)
    unet.eval()
    return cfg, unet


def load_vae(device: torch.device) -> AutoencoderKL:
    candidates = [
        ("stabilityai/sd-vae-ft-mse", None),
        ("runwayml/stable-diffusion-v1-5", "vae"),
    ]
    last_error: Exception | None = None
    for model_id, subfolder in candidates:
        try:
            kwargs = {"subfolder": subfolder} if subfolder else {}
            vae = AutoencoderKL.from_pretrained(model_id, **kwargs)
            vae.to(device)
            vae.requires_grad_(False)
            vae.eval()
            return vae
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Unable to load a Stable Diffusion VAE: {last_error}")


def make_content_dataset(repo_path: Path, style_dir: Path):
    ensure_repo_importable(repo_path)
    from data_loader.IAMDataset import IAMGenerateDataset

    with working_directory(repo_path):
        return IAMGenerateDataset(str(style_dir), "test", ref_num=1)


def normalize_text(text: str) -> str:
    return "".join(char if char in IAM_LETTERS else " " for char in text)


def load_style_tensor(style_image_path: Path, device: torch.device, min_width: int = 512, max_width: int = 1024) -> torch.Tensor:
    image = Image.open(style_image_path).convert("L")
    target_height = 64
    scaled_width = max(1, round(image.width * target_height / image.height))
    image = image.resize((scaled_width, target_height), Image.Resampling.LANCZOS)

    if image.width > max_width:
        image = image.crop((0, 0, max_width, target_height))
    if image.width < min_width:
        canvas = Image.new("L", (min_width, target_height), 255)
        canvas.paste(image, (0, 0))
        image = canvas

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def save_grayscale(tensor: torch.Tensor, output_path: Path) -> None:
    image = tensor.detach().cpu()
    if image.shape[0] >= 3:
        image = image[:3]
    pil_image = to_pil_image(image).convert("L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DiffBrushSingleRunner:
    def __init__(self, repo_path: Path, checkpoint_path: Path, style_dir: Path, device: torch.device) -> None:
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.style_dir = style_dir
        self.device = device
        self.cfg, self.unet = build_unet(repo_path, checkpoint_path, device)
        self.vae = load_vae(device)
        self.dataset = make_content_dataset(repo_path, style_dir)

        from models.diffusion import Diffusion

        self.diffusion = Diffusion(device=device)

    def generate_one(self, text: str, style_image_path: Path, output_path: Path, sampling_timesteps: int, eta: float = 0.0) -> None:
        normalized_text = normalize_text(text)
        if not normalized_text.strip():
            raise ValueError(f"DiffBrush target contains no supported characters: {text!r}")

        with working_directory(self.repo_path), torch.inference_mode():
            style_input = load_style_tensor(style_image_path, self.device)
            text_ref = self.dataset.get_content(normalized_text).to(self.device)
            latent = torch.randn(
                (1, 4, style_input.shape[2] // 8, self.dataset.fixed_len // 8),
                device=self.device,
            )
            generated = self.diffusion.ddim_sample(
                self.unet,
                self.vae,
                1,
                latent,
                style_input,
                text_ref,
                sampling_timesteps=sampling_timesteps,
                eta=eta,
            )
        save_grayscale(generated[0], output_path)


def find_style_dir(style_image: Path) -> Path:
    parent = style_image.parent
    if any(parent.glob("*.png")):
        return parent
    raise FileNotFoundError(f"No PNG style references found alongside style image: {style_image}")


def run_once(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    style_image = Path(args.style_image).resolve()
    out_dir = Path(args.out_dir).resolve()
    style_dir = find_style_dir(style_image)

    if not repo_root.exists():
        raise FileNotFoundError(f"DiffBrush repo not found: {repo_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"DiffBrush checkpoint not found: {checkpoint}")
    if not style_image.exists():
        raise FileNotFoundError(f"Style image not found: {style_image}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample.png"
    result_path = out_dir / "result.json"
    seed_everything(args.seed)
    runner = DiffBrushSingleRunner(repo_root, checkpoint, style_dir, device)
    runner.generate_one(
        args.prompt,
        style_image,
        sample_path,
        sampling_timesteps=args.steps,
        eta=float(args.eta),
    )
    result = {
        "prompt": args.prompt,
        "prompt_id": args.prompt_id,
        "run_id": args.run_id,
        "writer_id": args.writer_id,
        "conditioning_mode": args.conditioning_mode,
        "seed": args.seed,
        "steps": args.steps,
        "device": device.type,
        "repo_root": str(repo_root),
        "checkpoint": str(checkpoint),
        "style_image": str(style_image),
        "outputs": {
            "sample": str(sample_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one DiffBrush sample for handgen.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", default=str(REPO_DEFAULT))
    parser.add_argument("--style-image", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conditioning-mode", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eta", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    requested_device = select_device(args.device)
    try:
        run_once(args, requested_device)
    except Exception:
        if requested_device.type != "mps":
            raise
        gc.collect()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        run_once(args, torch.device("cpu"))


if __name__ == "__main__":
    main()
