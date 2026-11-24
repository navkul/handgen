# Local model setup

The `samples` route requires the base Handgen install and `rsvg-convert`. The following setup adds generation, OCR and writer adaptation. Run the commands from the repo root inside its virtual environment.

```bash
brew install librsvg tesseract
python -m pip install -e '.[diffbrush,vatr,vision,test]'
```

The Vision extra installs the macOS framework binding on macOS. On Linux, install the distribution packages that supply `rsvg-convert` and `tesseract`. Worksheet PDFs also use `pdftoppm` from Poppler.

## DiffBrush line generator

Download the authors’ pretrained file and save it as `models/diffbrush/third_party_repo/model_zoo/DiffBrush-ckpt.pt`. The inference code and character archetypes are in the vendored repository. Handgen reads the enrolled writer’s images and your text file. The upstream IAM benchmark images and WikiText prompt file can be obtained separately to run the authors’ benchmark. [Sources and licenses](../docs/references.md) [Authors’ checkpoint](https://drive.google.com/file/d/1EWzBmLtnQ42cTf3k_CYQ-nF3RXCb35I6/view) · [Upstream repository](https://github.com/dailenson/DiffBrush)

The checkpoint used for the reported measurements has SHA-256 `ed0d6059a3b0a68c18476c02ba2410d607fc478eaead597faa2483406c0082fa`.

The decoder checks the Hugging Face cache for the base VAE before attempting a download. To prepare an offline installation in advance, cache the decoder with the Hugging Face CLI. The writer’s reference stays on the local machine. [Decoder model](https://huggingface.co/stabilityai/sd-vae-ft-mse)

```bash
hf download stabilityai/sd-vae-ft-mse --include config.json diffusion_pytorch_model.safetensors
```

`DIFFBRUSH_ROOT` and `DIFFBRUSH_CHECKPOINT` override the code and checkpoint paths. `DIFFBRUSH_DEVICE=auto` selects CUDA, MPS or CPU. `DIFFBRUSH_STEPS` defaults to 20 and `HANDGEN_DIFFBRUSH_PROSE_CANDIDATES` to four regular attempts. Strict document rendering is available with `--require-verified`.

## VATr++ word generator and writer fitting

Handgen loads the generator from a safetensors conversion of VATr++. Download the pinned revision and check its SHA-256 before installing it. The model file is about 535 MiB. [Pinned files](https://huggingface.co/blowing-up-groundhogs/vatrpp/tree/fb73be2be8e96ce70d9cad9675a30667e19ac376) · [Authors’ training implementation](https://github.com/EDM-Research/VATr-pp)

```bash
mkdir -p models/vatrpp
curl --fail --location --retry 3 \
  https://huggingface.co/blowing-up-groundhogs/vatrpp/resolve/fb73be2be8e96ce70d9cad9675a30667e19ac376/model.safetensors \
  --output models/vatrpp/download.tmp
echo '3b1e4b7cae23652acd5c559117d06ef42fdd5317da2a5e0bc94ea44d8c0eb1ff  models/vatrpp/download.tmp' | shasum -a 256 -c - \
  && mv models/vatrpp/download.tmp models/vatrpp/model.safetensors
```

`HANDGEN_VATR_CHECKPOINT` and `HANDGEN_VATR_DEVICE` override the local path and device. `style fit` writes a reusable writer memory tied to the writer ID and base checkpoint digest. Pass the file with `--prose vatr --adapter PATH` to generate all prose with the fitted word model, or with `--adapter PATH` to use it for word repairs in automatic rendering. The base weights stay frozen during fitting.

## Handwriting recognition

Apple Vision runs locally on macOS with the Vision extra. Tesseract supplies a portable reader and word boxes. For the optional CPU TrOCR reader, install the DiffBrush extra above or install `torch torchvision transformers safetensors`, then download the following snapshot. [Microsoft’s model card](https://huggingface.co/microsoft/trocr-base-handwritten)

```bash
hf download microsoft/trocr-base-handwritten \
  --revision eaacaf452b06415df8f10bb6fad3a4c11e609406 \
  --exclude '*.bin' --local-dir .cache/trocr-base-handwritten
```

With both handwriting readers installed, `HANDGEN_OCR_ENGINE=auto` uses TrOCR for line crops at least 12 times wider than tall and Vision for shorter crops and words. With only TrOCR available it handles both line and word reads. Word-box detection uses Vision or Tesseract. Select a reader explicitly with `HANDGEN_OCR_ENGINE=vision`, `trocr` or `tesseract`. `HANDGEN_TROCR_MODEL` may point to an alternative local snapshot.

Set `HF_HUB_OFFLINE=1` after caching weights to run without model downloads. Weight files, the local catalog, and generated jobs are ignored by Git.
