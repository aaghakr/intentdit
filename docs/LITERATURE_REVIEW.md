# Layout-generation literature review and experiment requirements

## Scope

This review covers all 42 PDFs in `layout_related_papers/`. The core layout and
poster papers were read for their task definitions, datasets, conditioning
protocols, metrics, baselines, ablations, human studies, and limitations. Papers
on layered design, text rendering, document understanding, and image-generation
backbones were reviewed for evaluation practices that transfer to IntentDiT.

The directory also contains two non-paper items: `REPORT-md.pdf` is a project
report for LayoutWorld, and the ICLR workshop proposal is not a layout-generation
study. They are recorded below but should not be treated as scientific baselines.

## Main conclusion for IntentDiT

The present draft has a credible model idea and a useful backbone-by-conditioning
ablation, but the current evidence is not yet a strong journal evaluation. The
highest-priority deficiencies are:

1. The main tables compare image-only baselines with an IntentDiT variant that
   additionally receives prompts derived from ground-truth annotations. These
   protocols must not be ranked in one table.
2. There is no directly comparable text-conditioned baseline.
3. The evaluator does not report several established metrics: Alignment,
   Utilization, Small-element rate, layout FD/FID, and ground-truth/maximum IoU.
4. Prompt evaluation is dominated by template-derived counts. It needs independent
   human-written prompts, relation and hierarchy constraints, invalid-output rate,
   and a clear constraint-satisfaction score.
5. There is no completed human study, sampling-efficiency analysis, diversity
   analysis, or cross-dataset generalization experiment.
6. The paper says some results are over inference seeds, while the numbered
   scripts train independent models and use a fixed inference seed. The statistical
   unit and wording must be made consistent.

## What the literature actually measures

### Geometric and distributional layout quality

Generic layout work converges on four complementary measurements:

- **Layout FD/FID:** distributional similarity in a layout-feature space. It is
  used by LayoutDM, LayoutFormer++, LayoutFlow, LayoutNUWA, LACE, LayoutPrompter,
  Parse-Then-Place, TextLap, and Visual Layout Composer. Results are comparable
  only when the feature extractor, split, preprocessing, and sample count match.
- **MaxIoU/mIoU:** optimal matching between generated and reference layouts. Its
  definition changes across papers: identical category multisets, the same prompt,
  semantic text matching, or direct paired ground truth. The paper must state the
  matching population and assignment algorithm.
- **Alignment:** whether element edges and centers share design axes. In generic
  generation, the desirable value is often the value of real data, not necessarily
  zero. In poster benchmarks it is conventionally reported as lower-is-better.
- **Overlap:** unintended overlap. Underlays must be excluded because their
  intended function is to contain or overlap foreground content.

FID and MaxIoU capture fidelity/distributional behavior; Alignment and Overlap are
local geometric sanity checks. No one of these is a substitute for the others.

### Poster-specific graphic and content-aware quality

PosterLayout, LayoutPrompter, PosterLlama, LayoutDiT, PosterO, PosterLLaVa,
Desigen, CAL-RAG, and LaySPA motivate the following suite:

- **Validity (Val):** proportion of non-degenerate elements, normally using the
  0.1% canvas-area threshold.
- **Overlap (Ove):** overlap among non-underlay elements.
- **Alignment (Ali):** edge/center misalignment.
- **Underlay effectiveness (UndL/UndS):** loose intersection and strict
  containment of foreground elements by underlays.
- **Utilization (Uti):** use of non-salient space.
- **Occlusion (Occ):** conflict between layout elements and salient image regions.
- **Readability (Rea):** image-gradient/non-flatness under plain text regions.
- **Small-element rate (Sma):** LayoutDiT defines small elements using area below
  0.1% of the canvas or width/height below 2%.
- **Visual balance (VB):** used by PosterLLaVa/HPCVTG to detect unbalanced mass.

These metrics have different normalizations across repositories. IntentDiT must
validate its implementation against the official LayoutDiT/PosterLayout evaluator
on the same saved predictions before copying published baseline values.

### Controllability and prompt adherence

The strongest protocols do not reduce control to one aggregate count score:

- Parse-Then-Place separates type, position/size, and hierarchy consistency.
- LayoutFormer++ and LayoutPrompter report constraint violation rates.
- TextLap reports precision, recall, F-score, and invalid-format failure rate.
- PosterLLaVa measures user-constraint violation and uses explicit spatial
  requirements.
- SKE-Layout evaluates numerical/count reasoning separately from spatial
  reasoning and reports downstream image grounding success.
- PosterO introduces intent coverage and intent conflict.
- CreatiPoster, COLE/OpenCOLE, and BizGen explicitly judge prompt compliance or
  prompt following in addition to visual quality.

IntentDiT should therefore report count/type precision, recall, and F1; coarse
absolute-position accuracy; pairwise relation satisfaction; hierarchy/containment
satisfaction; invalid-output rate; and an overall constraint satisfaction score.
The current PLA can remain, but it must be described as count similarity rather
than full prompt-layout alignment. The current spatial PLA is requested-cell
recall, not a complete spatial-reasoning metric.

### Diversity

Diffusion and sampling-based systems should demonstrate that they can produce
multiple good layouts for the same condition. Relevant practices include layout
FID, Unique Match/DocSim in Parse-Then-Place, multiple generations per prompt in
LayoutPrompter, quality/diversity human ratings in LayoutNUWA, and qualitative
seed sweeps in COLE. Report both diversity and quality so random or poor layouts
cannot appear favorable.

For IntentDiT, generate at least 5 layouts for each of a fixed prompt/image subset
and report pairwise matched-box displacement or DocSim diversity, duplicate rate,
and the normal quality/adherence metrics. Include a seed grid for representative
examples.

### Human and learned-judge evaluation

Human studies are used because geometric metrics do not fully represent design
quality. The papers evaluate several distinct concepts: professional/aesthetic
preference, eligibility, prompt compliance, diversity, typography/readability, and
editability. These should not be collapsed into one vague “better layout” question.
IntentDiT needs a blinded, randomized study with at least two judgments:

1. visual professionalism/composition quality; and
2. agreement with the displayed user instruction.

Include a tie option, randomize left/right order, disclose participant count and
background, and analyze repeated ratings with participant/item-aware uncertainty.
The existing crossed participant/item bootstrap is appropriate. A minimum target
is 20 participants and 30 items per participant, with confidence intervals.

AesthetiQ, COLE/OpenCOLE, Accordion, CreatiPoster, and BizGen use MLLM judges.
If one is added, repeat stochastic judgments, randomize pair order, keep the rubric
fixed, and validate agreement against human ratings. An MLLM score must be a
supplement, not the only evidence.

### Efficiency and reproducibility

LayoutFlow, LayoutDiT, PixArt, SANA, BizGen, CreatiLayout, and Z-Image show that
quality should be considered together with model size and sampling cost. Report:

- trainable and total parameters;
- training GPU type/count, epochs, wall-clock time, and peak memory;
- inference steps, batch size, single-sample latency, throughput, and peak memory;
- a quality-versus-DDIM-steps curve, for example 10/20/50/100 steps.

Use identical hardware and warm-up when comparing latency. Archive per-image
predictions and exact environment/configuration files.

## Required experimental protocol

### Protocol A: fair image-only comparison

Inputs: background image and exactly the same image-derived signals available to
each method. No prompt, ground-truth count, coarse position, or test annotation may
be provided to IntentDiT.

Baselines should include at minimum LayoutDiT and RALF; add DS-GAN/CGL-GAN or
PosterLayout results only when preprocessing and evaluation are exactly matched.

Report Val, Ove, Ali, UndL, UndS, Uti, Occ, Rea, Sma, layout FD/FID, and paired
GT IoU/MaxIoU where defined. This must be the main quality table.

### Protocol B: image plus user instruction

Inputs: background image and the same natural-language instruction for all methods.
Separate annotation-derived template prompts from independently human-written
prompts.

Use at least one directly comparable text-capable baseline. Preferred order:

1. PosterLLaVa or PosterLlama with released model/code and the same test split;
2. a reproduced LayoutPrompter using the same prompt and image/saliency summary;
3. an open LLM structured-output baseline fine-tuned or prompted on the same data.

Report all quality metrics plus type/count P/R/F1, absolute-position satisfaction,
pairwise-relation satisfaction, hierarchy/containment satisfaction, invalid-output
rate, and human instruction preference. This is the controllability table.

### Protocol C: oracle diagnostic, not a headline comparison

Prompts derived from test annotations are useful for controlled diagnosis, but they
must be labeled **oracle annotation-derived constraints**. They answer whether the
model follows a known structured request, not whether it understands real user
intent. Keep these results out of the fair main baseline table.

## Minimum experiment set for the journal submission

### Must complete before submission

- [ ] Re-run the image-only comparison with actual LayoutDiT predictions under the
      same split, preprocessing, evaluator, sample count, and seeds.
- [ ] Add one genuine text-conditioned baseline to the instruction protocol.
- [x] Add Ali, Uti, Sma, paired IoU, category-multiset MaxIoU, visual balance,
      spacing, and a clearly labeled diagnostic layout FD to the evaluator.
- [ ] Produce layout FID with the official published feature extractor/checkpoint.
- [ ] Validate Val/Ove/UndL/UndS/Occ/Rea numerically against official evaluator
      outputs and document exact equations and aggregation rules.
- [ ] Run three independent training seeds and paired per-image bootstrap confidence
      intervals. Do not describe these as inference seeds.
- [ ] Resolve and archive the provenance of the CGL full-model checkpoint/results.
- [ ] Expand independent free-form evaluation to at least 100 prompts per dataset,
      written without looking at ground-truth boxes.
- [x] Implement relative-relation, containment/hierarchy, count, synonym, OOD,
      and contradictory-request stress evaluation.
- [ ] Run the implemented relation evaluation on final checkpoints.
- [ ] Complete the blinded human study for quality and instruction adherence.
- [x] Implement parameter, latency, peak-memory, throughput, and DDIM-step profiling.
- [x] Implement PKU-to-CGL and CGL-to-PKU cross-dataset tests, following the generalization
      emphasis of LayoutDiT and PosterO.
- [x] Implement same-condition diversity evaluation; final outputs still need running.
- [ ] Replace all provisional or provenance-uncertain values in the manuscript.

### Core ablations

- [ ] Image only / saliency / intent / saliency+intent.
- [ ] Text only / image+text / intent+text / saliency+intent+text.
- [x] Token-level cross-attention versus pooled text conditioning.
- [x] Intent pixel map versus intent boxes versus both.
- [x] Predicted intent map versus an explicitly labeled oracle GT-density diagnostic.
- [x] Loss isolation: no auxiliary losses, text loss only, intent loss only, default,
      and at least one higher setting for each loss.
- [ ] Sampling steps and guidance strength.
- [ ] Prompt styles: templated, paraphrased, independently written, spatial, and OOD.

The numbered scripts now include zero-loss, single-loss, default, and high-weight
conditions, plus token/pooled and pixel/box conditioning ablations.

### Valuable journal extensions

- [x] Element-complexity strata: 1--3, 4--6, and 7+ elements.
- [x] Saliency-complexity strata.
- [x] Prompt-edit locality: change one instruction and measure whether unrelated
      elements remain stable.
- [ ] Calibration curve between intent-map confidence and layout success.
- [ ] Human versus learned-judge correlation if an aesthetic judge is used.
- [ ] Visual balance and text contrast/readability beyond the gradient proxy.

## Current evaluator audit

The literature review reveals the following concrete issues in
`code/utils/metric.py`:

- Alignment and Utilization are implemented and reported per image and in aggregate.
- The framework now computes visual balance, paired IoU, category-multiset MaxIoU,
  small-element rate, out-of-bounds rate, and a clearly labeled handcrafted
  diagnostic FD. Official-feature layout FID remains an external evidence input.
- Validity is separated from a pre-clamping out-of-bounds metric.
- Occlusion and readability definitions must be checked against the official
  benchmark implementation. Published values are unsafe if normalizations differ.
- Readability now handles flat images without division by zero.
- Underlay and overlap define empty-layout behavior and omit non-evaluable values
  from aggregate means while retaining them in per-image records.
- The code calls the count score `tla`, while the paper calls it PLA. Use one name
  and state clearly that it measures symmetric count similarity only.
- Prompt templates and the evaluator share vocabulary and parsing rules. Report
  performance separately on independently written prompts to avoid a closed-loop
  metric.

## Annotated inventory of all files

### Core structured-layout generation

| File/paper | Main transferable lesson |
|---|---|
| Parse-Then-Place | Separates language parsing from placement; evaluates FID, alignment, overlap, MaxIoU, Unique Match, type/position-size/hierarchy consistency, diversity, ablations, and users. |
| LayoutFormer++ | Serializes heterogeneous constraints; reports mIoU, FID, alignment, overlap, and explicit violation rate over six tasks. |
| LayoutDM | Unified controllable discrete diffusion; uses matched preprocessing, three independent trials, FID and MaxIoU across six tasks. |
| LACE / aligned layout diffusion | Adds differentiable aesthetic constraints; uses FID, MaxIoU, alignment, overlap, post-processing ablations, and shared RICO/PubLayNet splits. |
| LayoutFlow | Flow matching for layouts; stresses real-data reference values, public-code retraining, multiple tasks, parameter count, and speed-quality trade-offs. |
| LayoutNUWA | Code-generating LLM for layouts; evaluates three datasets/tasks with FID, mIoU, alignment/overlap, domain transfer, ablations, and quality/diversity humans. |
| LayoutPrompter | Training-free LLM prompting/ranking; covers seven tasks and adds constraint/type/position-size violations to standard layout and poster metrics. |
| TextLap | Text-to-layout planning for open-set objects and visual text; adds P/R/F1 and invalid-format rate and redefines MaxIoU for same-prompt semantic matching. |
| Visual Layout Composer | Uses visual element content in vector/image diffusion; reports box/composed-image/generated-image FID and MaxIoU and discusses metric bias. |
| SKE-Layout | Spatial-knowledge retrieval for layout reasoning; separates numerical and spatial success and validates layouts through downstream simulation/image generation. |

### Content-aware poster layout

| File/paper | Main transferable lesson |
|---|---|
| LayoutDiT | Closest architectural baseline; uses PKU/CGL, graphic/content metrics, small-element analysis, constrained generation, cross-dataset tests, step ablation, and user preference. |
| PosterLlama | Multimodal LLM layout generation; reports Val/Ali/Ove/Und/FD/Occ/Rea and warns that inpaint artifacts can leak ground-truth locations. |
| PosterLLaVa | Unified multimodal/user-constrained generation; adds image FID, box IoU, visual balance, constraint violation, GPT evaluation, and human study. |
| PosterO | Layout trees and design-intent retrieval; introduces intent coverage/conflict, generalized datasets, multiple LLMs, retrieval ablations, and efficiency. |
| AesthetiQ | Preference-aligned MLLM layout prediction; pairs IoU with learned aesthetic win rate and validates judge-human correlation. |
| LLMs as Layout Designers / LaySPA | Reward-guided LLM spatial reasoning; reports format, collision, alignment, spacing, distribution, Ove/Und/Occ, and data efficiency. |
| CAL-RAG | Retrieval and agentic refinement on PKU; reports Ove/Ali/UndL/UndS and component ablations, but lacks content-aware and distributional metrics. |
| Desigen | End-to-end design template pipeline; motivates Salient Ratio, background FID/CLIP, layout Ali/Ove/Occ, iterative refinement, and users. |

### Full graphic/poster generation and layered design

| File/paper | Main transferable lesson |
|---|---|
| COLE | Hierarchical editable design generation; uses a broad intention benchmark, GPT-4V dimensions, general/professional users, IoU placement, seed diversity, and component ablations. |
| OpenCOLE | Reproducible COLE implementation; evaluates design/layout, relevance, typography/color, graphics/images, and innovation and exposes reproducibility gaps. |
| CreatiPoster | Editable multimodal poster pipeline; separates layout, color, graphic style, and compliance; uses repeated MLLM judgments and blinded humans. |
| Accordion / Rethinking Layered Graphic Design | Converts generated raster designs to editable layers; evaluates five design dimensions, aesthetics, layer-count error, editability with designers, and task ablations. |
| LayerD | Raster-to-layer decomposition; introduces edit-aware sequence matching, RGB L1/alpha soft-IoU, ablations, and a 21-worker user study. |
| LICA | Large structured layered-design dataset; explains why geometric metrics and uncalibrated LLM judges are insufficient and motivates category-conditioned learned evaluation. |
| BizGen | Dense infographic/slide generation; reports global and region-wise prompt following, OCR, layer success, style consistency, latency, users, and scaling ablations. |
| POSTA | Customized artistic poster generation; emphasizes editable assets, OCR precision/recall, semantic/style/layout quality, and customization. |
| CreatiLayout | Layout-to-image and layout planning; evaluates region spatial/color/texture/shape adherence, global image quality, format accuracy, and downstream rationality. |

### Adjacent image composition, text rendering, and foundation models

| File/paper | Relevance to IntentDiT |
|---|---|
| Multitwine | Demonstrates global and crop-level prompt/object fidelity and identity leakage tests; useful if layouts are rendered downstream. |
| Glyph-ByT5 and Glyph-ByT5-v2 | Show that text accuracy and aesthetic quality need separate OCR and human measurements; useful for future full-poster rendering. |
| DreamPainter | Uses text/reference fidelity metrics and pairwise humans for e-commerce inpainting; relevant to background consistency, not layout boxes. |
| `CreatiDesign...pdf` | The extracted document content describes the DreamPainter-style e-commerce background task; verify the file/title before citing it. |
| PixArt-alpha and PixArt-delta | Transfer the practice of reporting FID/alignment, user preference, model size, training cost, sampling steps, and control ablations. |
| SANA | Strong efficiency reporting: parameters, reconstruction/generation metrics, latency, throughput, memory, and quality-versus-steps. |
| Z-Image | Broad prompt-following/text benchmarks plus latency/throughput; useful as an evaluation-design reference, not a layout baseline. |
| Pix2Struct | Screenshot parsing pretraining; relevant to structured visual encoding but not a layout-generation baseline. |
| ScreenAI | UI/infographic understanding; supports OCR/no-OCR and model-scale ablation practices, not generation comparison. |
| Simple Layout Token / LayTokenLLM | Document understanding with layout tokens; supports spatial representation and scale/ablation ideas, not poster generation. |
| Graph diffusion survey | Useful theoretical taxonomy and caution about graph generative metrics; not an experimental poster baseline. |

### Non-baseline files

| File | Treatment |
|---|---|
| `REPORT-md.pdf` | Internal LayoutWorld/MCTS project report. It offers reward-shaping and search lessons but is not a peer-reviewed baseline. |
| ICLR 2025 workshop proposal | Not a layout-generation paper; exclude from related work and experiments. |

## Recommended order of work

1. Repair and validate the evaluator against official code.
2. Freeze fair image-only and instruction-conditioned protocols.
3. Integrate one text-conditioned baseline and reproduce LayoutDiT predictions.
4. Run the missing loss/architecture ablations and three training seeds.
5. Run free-form, relation, OOD, diversity, efficiency, and cross-dataset tests.
6. Generate final renders and conduct the two-question human study.
7. Aggregate paired uncertainty, archive provenance, and only then update all paper
   tables, claims, related work, and limitations.
