# Handgen

Handgen is a local research prototype for few-shot handwriting synthesis. Given a few transcribed lines, it can generate words a writer never supplied, compose inline math from the writer’s worksheet, and trace each mark on the exported page back to its source.

![Five handwritten lines explaining why 2n + 1 is odd for every integer n.](docs/assets/odd-integer-proof.png)

For this proof, a fitted VATr++ model generated all 25 English word placements. Four math spans use the writer’s worksheet. The inputs were three handwritten lines and one symbol sheet. A blind OCR read of the finished page found two edits across 118 normalized prose characters. [Typed proof](examples/odd_integer_proof.txt) · [Actual handwriting samples used](docs/assets/odd-integer-sources.png) · [Recorded measurements](docs/results.json)

## Try it with the bundled samples

The first example runs on CPU and takes nine words from a real handwritten pangram and rearranges them into a new line. Install Python 3.11 or newer and the SVG rasterizer. On macOS, run `brew install librsvg`. On Debian or Ubuntu, install `librsvg2-bin`.

```bash
git clone https://github.com/navkul/handgen.git
cd handgen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export HANDGEN_DATA_DIR="$PWD/outputs/catalog"

handgen init
handgen writer create writer_002
handgen style add writer_002 data/style_refs/writer_002/natural_prose_primary.png \
  --transcript examples/transcripts/writer_002_primary.txt
handgen document render writer_002 examples/recombined_prose.txt \
  --prose samples --require-verified --out outputs/words
```

Open `outputs/words/document.png`. The same folder contains the SVG, the exact input, and a manifest with the source range and evidence for every placement. Run `python -m pytest -q` for the test suite. [Original pangram, rearranged words and a mixed proof](docs/assets/handgen-preview.png)

## Fit a writer and generate new text

This example uses a second writer. It first enrolls three transcribed lines and a symbol sheet. You can render the worksheet example immediately with the base install.

```bash
handgen writer create writer_001
for ref in primary alt01 alt04; do
  handgen style add writer_001 "data/style_refs/writer_001/natural_prose_$ref.png" \
    --transcript "examples/transcripts/writer_001_$ref.txt"
done
handgen worksheet ingest writer_001 data/worksheets/writer_001/writer_001_worksheet_01.png
handgen document render writer_001 examples/mixed_proof.txt \
  --prose samples --require-verified --out outputs/worksheet-proof
```

For neural generation, install the VATr++ checkpoint and a local OCR reader. The model guide includes download revisions, SHA-256 digests, and Apple Silicon setup. An M5 Pro ran the reported fit. [Model setup](models/README.md#vatr-word-generator-and-writer-fitting)

```bash
handgen style fit writer_001 --steps 100 --seed 732 --out outputs/fit \
  --test-reference data/style_refs/writer_001/natural_prose_alt03.png \
                   examples/transcripts/writer_001_holdout.txt
handgen eval render writer_001 --source examples/odd_integer_proof.txt \
  --prose vatr --adapter outputs/fit/writer.safetensors --out outputs/odd-proof
```

Open `outputs/odd-proof/render/document.png` to see the proof and `outputs/odd-proof/metrics.json` to inspect the page readback. To generate your own note, pass a UTF-8 text file and mark inline math with double brackets, for example `Choose [[x ≥ 0]].`

## Engineering and evaluation

| Component | Implementation and evidence |
| --- | --- |
| Writer adaptation | Freeze VATr++ and fit a 512-parameter offset to its style representation. Hold out word types for checkpoint selection and a separate handwritten line for testing. On five previously unseen words from writer_001, mean shape loss fell from 0.4385 to 0.4188. Three words improved and two regressed. |
| Prose generation | DiffBrush generates short lines. A dynamic-programming chunker can choose word boundaries. Automatic rendering can reuse transcribed words, keep readable parts of a generated line and fill gaps with VATr++ words or worksheet letters. |
| Math and provenance | Segment a labeled worksheet into a glyph bank, then align handwritten symbols with the prose baseline. Save original input bytes, visible text, source offsets, rendering method and verification evidence alongside the SVG and PNG pages. |
| Page evaluation | Read prose back from exported PNG regions without supplying the target to the recognizer. On one 124-character note, DiffBrush had 16 OCR edits, base VATr++ had four, fitted VATr++ had one and the automatic route with a worksheet had zero. |

The shape metric measures normalized ink overlap and density. The page metric measures what an OCR reader can recover. For the five held-out words, the comparison image shows the actual ink beside both model outputs. [Held-out word comparison](docs/assets/handgen-writer-fit.png) · [Experiment setup, failure cases and reproduction commands](docs/findings.md) · [Machine-readable results](docs/results.json)

Use `--prose samples` for sample and worksheet composition, `--prose diffbrush` for line generation, `--prose vatr` for word generation, or `--prose auto` for the combined route. Add `--require-verified` to require a transcript label, a worksheet label or an exact normalized OCR match for every visible placement. [CLI and file formats](docs/pipeline.md) · [Architecture](docs/architecture.md) · [Design decisions](docs/decisions.md)

## Contributors and citation

**Arnav Kulkarni and Felix Tuchscherer are the co-authors of Handgen.** The development history includes the reusable inference process, source-aware rendering, worksheet registration fixes and the chunking benchmark. [Inference](https://github.com/navkul/handgen/pull/1) · [Rendering](https://github.com/navkul/handgen/pull/2) · [Worksheet registration](https://github.com/navkul/handgen/pull/3) · [Chunking and benchmarking](https://github.com/navkul/handgen/pull/4)

Cite both authors when referring to Handgen and cite the original research for the models you use. GitHub’s **Cite this repository** button provides a software citation. [Handgen citation](CITATION.cff) · [Research papers, checkpoints, datasets and software credits](docs/references.md)
