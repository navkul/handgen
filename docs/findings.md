# Findings

The accompanying JSON contains the recorded word splits, losses, page reads, evidence counts and checkpoint digests for the experiments below. [Download the measurements](results.json)

## Reading a finished page

The following comparison used the same writer_001 enrollment, 20 DiffBrush steps and the default render seed. The machine ran macOS 26.6.2 on an M5 Pro with Python 3.12.0, PyTorch 2.12.0, Diffusers 0.38.0, Transformers 5.8.1 and the PyObjC Vision 13.0 binding. Every route rendered the same three source lines from `examples/prose_only.txt`. The note wrapped into four physical prose regions. The evaluator read those regions from the final PNG with local Vision or TrOCR according to crop shape.

| Prose route | Page OCR edits / 124 compared characters | Exactly read regions / 4 | Visible placements lacking a label or exact crop read |
| --- | ---: | ---: | ---: |
| DiffBrush only | 16 | 1 | 4 |
| VATr++ with reference crops | 4 | 1 | 4 |
| VATr++ with the selected writer adapter | 1 | 3 | 2 |
| Automatic, with the writer’s worksheet | 0 | 4 | 0 |
| Automatic, with the worksheet and fitted adapter | 0 | 4 | 0 |

Automatic rendering retained six independently recognized generated portions and used worksheet lettering in seven placements. Adding the fitted adapter replaced five of those seven with OCR-matched VATr++ words. The two remaining worksheet repairs were “before” and “follows.” A crop-level OCR match and a page-level read answer different questions. Two word crops in the VATr++-only fitted run failed individually even though the complete page was only one edit away. Every source sidecar reconstructed its input exactly.

On the separate four-line mixed proof, the automatic route read back with two edits over 95 prose characters. With a fitted adapter it had one edit, reading “offet” for “offset.” The mixed-proof panel in the pangram comparison shows this fitted version. [Comparison image](assets/handgen-preview.png) Its three math expressions trace to worksheet samples. All nine visible placements have recorded support and visible ink. The writer_002 recombined pangram read back with zero edits over 43 characters and used nine transcribed words. A 72-line CRLF fixture with repeated equations and blank lines produced three distinct SVG/PNG pages and preserved every input byte.

The five-line odd-integer proof used the same three reference lines, writer adapter and symbol worksheet in both renders. The fitted word model generated all 25 prose word placements in the first render. Two crop readings carried warnings, and page OCR counted two edits over 118 normalized prose characters. [Typed input](../examples/odd_integer_proof.txt) · [VATr++ proof](assets/odd-integer-proof.png) · [Handwriting references](assets/odd-integer-sources.png)

In the combined render, DiffBrush provided ten prose segments. Fitted VATr++ supplied “remainder” and “odd.” on line four. The worksheet supplied “one.” on line three and all four math spans. Each visible placement had a label or a matching crop read. On the finished PNG, OCR read “quotient” as “quatient,” so the page missed the exact-readback criterion by one character out of 118. The combined run used seed 732 and the word-model run used seed 2026051301. These are two examples of the rendering routes. [Combined proof](assets/odd-integer-proof-combined.png) · [Placement and OCR records](results.json)

For writer_002, automatic rendering of the same novel 124-character prose note had only the pangram as a reference and no worksheet. The renderer combined a generated line, recovered crops and VATr++ words. The page had one OCR edit, and three words carried individual verification warnings.

Reproduce the prose comparison after enrollment and model setup with the following commands.

```bash
for mode in auto diffbrush vatr; do
  handgen eval render writer_001 --source examples/prose_only.txt \
    --prose "$mode" --out "outputs/eval-$mode"
done
handgen eval render writer_001 --source examples/prose_only.txt --prose vatr \
  --adapter outputs/fit/writer.safetensors --out outputs/eval-fitted
handgen eval render writer_001 --source examples/prose_only.txt --require-verified \
  --adapter outputs/fit/writer.safetensors --out outputs/eval-automatic-fitted
```

The local OCR configuration influences these measurements. Each metrics file keeps the recognized text, reader, source range via the manifest, region bounds and character edit count.

## Few-shot fitting on the M5 Pro

VATr++ fitting optimized 512 shared style-conditioning parameters with the base generator frozen. Both fits used 100 Adam steps, seed 732, and a validation check every ten steps. The baseline used only the training crops for its conditioning. Loss compares normalized ink shapes at three scales and includes an ink-density penalty. Lower values indicate closer agreement under this metric.

| Writer | Training word labels | Validation word labels | Selected step | Baseline loss | Selected loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| writer_001 | 14 | 6 | 70 | 0.4279 | 0.3662 |
| writer_002 | 7 | 2 | 10 | 0.2857 | 0.2544 |

Capitalization variants stay in the same split. Labels count case-sensitive word types, so `Brand` and `brand` count as two labels and one family. The validation loss is used to select the checkpoint. Writer_002 has only a single labeled pangram.

For writer_001, a separate reference from another original page provided five aligned alphabetic words absent from enrollment. That image was unavailable to fitting and step selection. Average test loss fell 4.5%, from 0.4385 to 0.4188.

| Independent word | Before | After |
| --- | ---: | ---: |
| our | 0.373 | 0.335 |
| business | 0.421 | 0.355 |
| will | 0.457 | 0.545 |
| should | 0.320 | 0.337 |
| Objectives | 0.621 | 0.523 |

![Actual writer samples and VATr++ output before and after fitting for the five independent words.](assets/handgen-writer-fit.png)

The result supports a modest average shape improvement on this five-word sample, with two regressions. A larger independent collection across writers and blinded human ratings would be needed to assess perceived resemblance.

## Learning from real ink

- Three transcribed lines supplied 26 word and punctuation crops for writer_001. A single pangram supplied nine word crops for writer_002. On a separate two-row page crop, enrollment and recombination produced a three-line note from actual word ink.
- Whole-page reference images need segmentation. On three scanned lecture pages the extractor found 30, 29 and 30 handwritten rows. On a fourth page, selecting the handwritten region yielded 18. Ruled lines, descenders and text from nearby rows often intersect on these pages. Geometry and segmentation choices remain inspectable.
- In two paired, same-seed, 20-step DiffBrush trials, conditioning on a recovered line reduced mean absolute normalized stroke-thickness deviation from 0.072 to 0.012 relative to 43 other windows of the same page. Neither arm had an exact Tesseract read, and one prepared-reference sample omitted the final word. This small test measured stroke thickness. It provides no writer-identification accuracy estimate. The measurement used skeletonized ink and body height. [scikit-image skeletonization](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize)
- On held-out word families, a regularized character-class width model reduced writer_001’s mean absolute percentage error from 15.6% to 8.9%. For writer_002, it increased error from 15.4% to 17.1%. The renderer enabled personalized width correction for the first writer and kept native word widths for the second. Interword spacing also uses actual observed gaps where available.

## Generation and recognition

- The reference DiffBrush canvas can encourage repetitions on very short prompts. Across six short prompts, two writers and two seeds, the length-conditioned canvas reduced blind Vision character edits from 538 to 101 over 300 target characters. Extra output counts as insertions, so edit totals can exceed the target length. In a separate 12-pair short-prompt check, generating context and extracting the prefix reduced 52 edits to 24.
- Sampling reuses the same encoded style and content on successive diffusion steps. On the M5 Pro the median of three warm 20-step samples per path fell from 0.742 to 0.596 seconds. Four paired comparisons produced byte-identical PNGs. This measures the sampling stage.
- From four rejected chunks of a mixed proof, clear gaps plus a fresh OCR pass recovered 12 out of 16 expected words. A crop is discarded if its letters touch rejected ink or if rereading changes the word. Whole lines sometimes contain false repetitions, omitted letters or punctuation.
- On 23 VATr++ word placements totaling 105 visible characters, Apple Vision read 19 exactly with seven edits, CPU TrOCR read 12 exactly with 13 edits, and Tesseract read none exactly with 73 edits. Median warm time across 19 distinct crops was 15.6 ms for Vision, 177.9 ms for three-thread CPU TrOCR and 40.2 ms for Tesseract on this machine. This sample of word crops does not establish a ranking on full lines or other writers.

## Worksheet and layout limits

The supplied worksheet produced 92 labeled crops. Ingestion recovered the same inventory from rotated variants through ±6°, shadowed pages and ruled variants. A direct label takes precedence over a composed symbol. The sheet lacks gamma. Strict rendering reports that gap for a source that needs it. The same policy covers custom labels and missing or damaged crops.

Touching glyphs without separable boundaries and symbols written out of the declared order require a corrected scan or layout. The compositor supports inline notation, common derived operators and Unicode script digits. Fractions written in two dimensions, matrices, nested typesetting and unconstrained equation recognition remain open work. Worksheet prose is readable, although isolated letters lose the joins and variation present in actual word samples. Repeated uses of a word can look identical when only one sample exists.
