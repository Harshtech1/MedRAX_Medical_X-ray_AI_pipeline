# MedRAX ChestAgentBench Reproducibility Audit

## Scope and conclusion

This audit compares the checked-out repository against the ChestAgentBench protocol described in *MedRAX: Medical Reasoning Agent for Chest X-ray* (arXiv:2502.02673v2, ICML 2025). The paper states that ChestAgentBench contains 2,500 six-choice questions, that model performance is measured by accuracy across all questions, and that unclear responses, errors, and timeouts are retried up to three times before being marked incorrect.

Before the approved fidelity fixes, the checked-in evaluation code could not guarantee reproduction of Table 1. The original blockers were:

1. `experiments/benchmark_medrax.ipynb` excludes every case with `case_id <= 17158`. Against the released `chestagentbench/metadata.jsonl`, this leaves only 298 of 2,500 questions.
2. The MedRAX notebook has no effective three-attempt retry or answer-validation loop, despite importing Tenacity.
3. `experiments/analyze_axes.py` divides by parsed predictions, excluding malformed, missing, and error outputs instead of counting them as incorrect.
4. The repository does not contain the `benchmark/questions/` tree consumed by the notebook. The downloaded Hugging Face representation is a single `chestagentbench/metadata.jsonl` file with a different schema.
5. The notebook reuses one LangGraph thread across all questions belonging to a case. The paper does not state whether benchmark questions were evaluated independently, so the exact protocol is ambiguous.
6. The paper used GPT-4o and an RTX 6000. The current local replication controller is `google/gemini-2.5-flash` through OpenRouter and the available GPU is a Tesla T4. That run can evaluate the framework but cannot reproduce the published MedRAX number.

The user approved the minimal protocol fixes. They are now applied to the paper-facing notebook and MedRAX aggregation path; prompts, model selection, tool selection, LangGraph architecture, benchmark ordering, and per-case memory behavior remain unchanged.

## Repository provenance and state

- Audited commit: `34fb87d0da91313712dd72097b462fda9634aa88`
- Configured origin: `https://github.com/Harshtech1/MedRAX_Medical_X-ray_AI_pipeline.git`, not the paper's canonical `https://github.com/bowang-lab/MedRAX` remote.
- Pre-existing local source changes were present in `main.py` and `medrax/tools/generation.py`. They were not created by this audit.
- The published Hugging Face metadata contains 2,500 unique question IDs from 609 cases, 4,629 image references, and one record with no image reference. All referenced local image files are present.
- The paper describes 675 curated source cases; fewer cases can remain after question generation and quality filtering, so 609 represented cases is not by itself evidence of a split.

## Evaluation pipeline and dependency graph

```text
Paper Table 1: ChestAgentBench
|
+-- Dataset construction (not evaluation)
|   +-- data/eurorad_metadata.json
|   +-- benchmark/create_benchmark.py
|       +-- benchmark/llm.py
|       +-- writes benchmark/questions/<case>/<question>.json
|
+-- MedRAX paper evaluation
|   +-- experiments/benchmark_medrax.ipynb
|       +-- data/eurorad_metadata.json
|       +-- benchmark/questions/**/*.json
|       +-- benchmark/figures/**
|       +-- medrax/docs/system_prompts.txt
|       +-- medrax/agent/agent.py
|       +-- medrax/tools/{classification,segmentation,grounding,
|       |                    report_generation,xray_vqa,llava_med}.py
|       +-- ChatOpenAI(model="gpt-4o", temperature=0.2, top_p=0.95)
|       +-- writes JSON-lines experiment log
|   +-- experiments/analyze_axes.py --model medrax
|       +-- parses answer letters
|       +-- computes overall/category/question-type accuracy
|       +-- output was manually summarized into Table 1
|
+-- Paper baselines
|   +-- experiments/benchmark_gpt4o.py
|   +-- experiments/benchmark_llama.py
|   +-- experiments/benchmark_chexagent.py
|   +-- experiments/benchmark_llavamed.py
|   +-- experiments/analyze_axes.py / experiments/compare_runs.py
|
+-- Convenience/demo paths (not the MedRAX Table 1 agent run)
    +-- quickstart.py                standalone OpenAI-compatible VLM evaluator
    +-- README command              two-sample smoke test via --max-cases 2
    +-- main.py                     Gradio application entry point
    +-- inspect_logs.py             log inspection only
    +-- validate_logs.py            missing/error diagnostics only
```

### File classification

| File | Role | Used for published MedRAX agent metric? |
|---|---|---|
| `experiments/benchmark_medrax.ipynb` | MedRAX agent benchmark runner and logger | Intended yes, but checked-in final cell is unexecuted and contains a resume filter |
| `experiments/analyze_axes.py` | Overall and axis-level metric aggregation | Intended yes; `--model medrax` routes to the GPT-4-format parser |
| `experiments/benchmark_gpt4o.py` | GPT-4o non-agent baseline | Baseline only |
| `experiments/benchmark_llama.py` | Llama-3.2-Vision baseline | Baseline only |
| `experiments/benchmark_chexagent.py` | CheXagent baseline | Baseline only |
| `experiments/benchmark_llavamed.py` | LLaVA-Med baseline/client | Baseline only; defaults behave as a two-case raw-output smoke test |
| `quickstart.py` | Standalone OpenAI-compatible VLM evaluation over HF JSONL | No; it never constructs the MedRAX agent or tools |
| `main.py` | Interactive Gradio application | No |
| `experiments/compare_runs.py` | Ad hoc comparison and common-question intersection | Not a faithful fixed-2,500 denominator aggregator |

The repository contains no committed raw result logs or script that directly regenerates the complete published Table 1. The table appears to require running each evaluator, aggregating logs, and manually collating the reported values.

## Complete benchmark-filter audit

### MedRAX paper-facing path

| Location | Filter or early exit | Effect |
|---|---|---|
| `benchmark_medrax.ipynb` | `if int(case_details["case_id"]) <= 17158: continue` | Critical: leaves 298/2,500 released questions |
| `benchmark_medrax.ipynb` | no question files for a case -> `continue` | Omits questions if the generated tree and Eurorad manifest disagree |
| `benchmark_medrax.ipynb` | no matching figure -> `continue` within figure matching | Drops that figure; the question can still run without the missing image |
| `benchmark_medrax.ipynb` | `final_response is None` -> skipped | Nominal skip path; current exception handler returns empty strings instead of `None` |
| `analyze_axes.py` | `--max-questions` -> `break` | Optional metric truncation |
| `analyze_axes.py` | HTTP lines and malformed JSON -> `continue` | Removes records from the denominator |
| `analyze_axes.py` | only increments total when model and reference letters parse | Critical: accuracy is correct/parsed, not correct/attempted |

### Demo and baseline paths

| Location | Filter or early exit | Effect |
|---|---|---|
| `quickstart.py` | `--max-cases` | Explicit subset; README demonstrates `--max-cases 2` |
| `quickstart.py` | shutdown event -> `break` | Partial run |
| `quickstart.py` | no usable image -> skipped | Removes sample from output/metric inputs |
| `benchmark_gpt4o.py` | missing question files or images -> skipped | Partial baseline denominator |
| `benchmark_llama.py` | missing question files or images -> skipped | Partial baseline denominator |
| `benchmark_chexagent.py` | missing question files/images or inference failure -> skipped | Partial baseline denominator |
| `benchmark_llavamed.py` | `raw_output=True` plus default `--num-cases 2` | Always stops after two cases unless configured otherwise |
| `benchmark_llavamed.py` | accuracy denominator subtracts skipped questions | Correct/processed rather than correct/all expected |
| `analyze_axes.py` | line slicing/breaks for `--max-questions` in all parsers | Optional subset |
| `compare_runs.py` | intersects question IDs across models | Reports only common successfully logged predictions |
| `compare_runs.py` | random sample of five | Console diagnostics only; does not limit its accuracy loop |
| `benchmark/create_benchmark.py` | `skip_first=100` and slice excluding the final 100 sorted cases | Dataset-generation filter, not evaluation; do not regenerate the published benchmark with defaults |
| `chexbench_gpt4.py` | selects the Visual Question Answering subset | Applies to CheXbench, not ChestAgentBench |

No benchmark-relevant use of `head()`, `tail()`, `shuffle`, or an unconditional list slice was found outside the optional limits and utilities above. `continue` statements inside medical tool postprocessing do not filter benchmark samples.

## Denominator audit

The paper requires:

```text
overall_accuracy = correct_predictions / 2500
```

The MedRAX branch in `analyze_axes.py` calls `analyze_gpt4_results`. That function increments `all_questions` only inside:

```python
if model_letter and correct_letter:
    all_questions += 1
```

It therefore computes:

```text
overall_accuracy = correct_predictions / parsed_predictions
```

Consequences:

- Invalid model responses are excluded.
- Error records lacking `model_answer`/`correct_answer` are excluded.
- Missing log records are invisible.
- Malformed JSON lines are excluded.
- If a retry implementation logs multiple attempts, duplicate question IDs can be counted more than once.
- Category denominators are likewise reduced because categories are incremented only for parsed predictions.

The checked-in code does not assert 2,500 unique question IDs and cannot guarantee a fixed denominator.

## Retry audit

The paper says unclear responses, errors, and timeouts are retried up to three times; unresolved samples are marked incorrect.

| Failure class | MedRAX notebook | Faithful? |
|---|---|---|
| API/transport exception | No explicit sample-level retry | No |
| Timeout | No explicit timeout or retry | No |
| Malformed tool/API JSON | No explicit retry; likely propagates to broad exception handler | No |
| Invalid answer letter | No validation before logging | No |
| Final failure | Logs an error without `model_answer` or `correct_answer`; analyzer excludes it | No |

The notebook imports `retry`, `wait_exponential`, and `stop_after_attempt` but never decorates a function or invokes a retry loop. `ChatOpenAI` may perform provider-level transport retries depending on the installed LangChain/OpenAI versions, but that is not equivalent to the paper's explicit three-attempt sample protocol and does not cover invalid answers.

Other scripts implement fragments of the paper policy:

- `quickstart.py` and `benchmark_gpt4o.py` use Tenacity with three attempts for raised exceptions, but do not validate/retry malformed answer choices.
- `benchmark_llavamed.py` retries network/stream errors three times, but skips no-image cases and its default run stops after two cases.
- `benchmark_llama.py` has transport retries and answer regexes, but persistent failures are subsequently skipped/excluded.
- `benchmark_chexagent.py` has answer extraction but no three-attempt inference retry.

No single checked-in evaluator implements all four required retry classes and then preserves final failures as incorrect.

## Answer-extraction audit

The paper says responses are processed with regex to extract a single letter choice, but does not publish the exact regex.

The parser actually selected for MedRAX is `extract_answer_letter` in `experiments/analyze_axes.py` because `--model medrax` calls `analyze_gpt4_results`. It is not regex-based. It accepts:

- any single alphabetic character, including values outside A-F;
- an alphabetic first character followed by `) . : -` or space;
- otherwise returns `None`.

It does not extract `B` from prose such as `The answer is B`, despite the paper describing regex extraction.

Other, non-MedRAX extraction implementations coexist:

- `compare_runs.py`: several increasingly permissive A-F regexes, ending with the first A-F character anywhere in the response.
- `benchmark_chexagent.py`: `re.search(r"([A-F])", cleaned)`.
- `benchmark_llama.py`: ordered patterns including `ANSWER:`, `OPTION`, `A)`, and standalone A-F; a second validator uses the first A-F character.
- `benchmark_llavamed.py`: strict letter first, then first A-F character.

Because the paper omits its exact regex and the repository contains multiple conflicting implementations, the exact published extraction rule cannot be proven from the release. The MedRAX metric path does not literally match the paper's regex description.

## Paper-to-repository comparison

| Paper | Repository | Match? |
|---|---|---|
| 2,500 six-choice ChestAgentBench questions | Published HF JSONL has 2,500 unique questions | Yes, dataset artifact |
| Evaluate accuracy across all questions | Notebook case cutoff leaves 298 released questions | **No** |
| Failures remain incorrect | Notebook/analyzer exclude persistent failures | **No** |
| Fixed denominator of 2,500 | Denominator is parsed predictions | **No** |
| Retry unclear responses, errors, and timeouts up to three times | No effective MedRAX retry loop | **No** |
| Regex extracts A-F choice | MedRAX analyzer uses a non-regex prefix parser | Partial/no |
| GPT-4o backbone | Notebook uses `ChatOpenAI(model="gpt-4o")` | Yes in notebook; current Gemini/OpenRouter replication is not equivalent |
| Temperature 0.2 | Notebook sets 0.2 | Yes |
| Top-p | Paper does not specify in main text; notebook uses 0.95 | Unverifiable |
| Published medical-assistant prompt | `medrax/docs/system_prompts.txt` matches Appendix A wording | Yes |
| Benchmark user prompt asks for careful reasoning and critical tool use | Notebook adds this prompt and a second letter-only turn | Repository-specific; consistent with paper, exact publication detail incomplete |
| VQA: CheXagent and LLaVA-Med | Both enabled in notebook | Yes |
| Grounding: MAIRA-2 | Enabled, 8-bit | Yes |
| Segmentation: ChestX-Det/MedSAM implementation | Enabled through segmentation tool | Yes |
| Classification: TorchXRayVision | Enabled | Yes |
| Report generation: CheXpert Plus model | Enabled | Yes |
| Image generation exists in architecture | RoentGen is not enabled in benchmark notebook | Benchmark implementation appears intentionally narrower |
| Utilities available in framework | DICOM and visualization are not bound in benchmark notebook | Consistent with benchmark notebook; differs from current eight-tool local run |
| ReAct loop bounded by `tmax` | LangGraph loop has no explicit time limit; framework recursion limit is implicit/version-dependent | **No/underspecified** |
| Tool failure returned to reasoning loop | uncaught tool exceptions abort the sample | **No** |
| Independent benchmark samples | One thread is reused for all questions in each case | Unspecified/ambiguous |
| Parallel or sequential tool calls | Multiple calls are executed sequentially in `Agent.execute_tools` | Partial; predictions may be unaffected, timing differs |
| Single RTX 6000 | Current environment is a Tesla T4 with CPU placement for large tools | No; primarily runtime/performance impact |
| Reproducible model artifacts | HF and TorchXRayVision model revisions are not pinned | **No** |
| Reproducible controller snapshot | Mutable `gpt-4o` alias, no dated snapshot/seed | **No** |

### Max turns

Neither the paper nor notebook specifies a numerical maximum number of ReAct turns. The paper's algorithm uses a maximum elapsed time `tmax`, while the repository relies on LangGraph's implicit recursion behavior and ends when the controller emits no tool calls. Therefore a numerical max-turn match cannot be verified.

### Tool set used by the paper-facing notebook

The notebook binds six tools:

1. Chest X-ray report generation
2. Chest X-ray classification
3. Chest X-ray segmentation
4. Phrase grounding (MAIRA-2)
5. CheXagent VQA
6. LLaVA-Med VQA

`ImageVisualizerTool`, `DicomProcessorTool`, and RoentGen are not part of that benchmark tool list. Binding eight tools in the local pilot is therefore not an exact reproduction of the paper-facing notebook.

## Deviations ranked by expected impact

| Rank | Deviation | Expected impact |
|---:|---|---|
| 1 | Different controller (`gemini-2.5-flash` vs GPT-4o) | Very high; changes vision, reasoning, tool selection, and answer synthesis |
| 2 | Case-ID cutoff evaluates 298 rather than 2,500 questions | Very high; invalidates overall and category metrics |
| 3 | Correct/parsed denominator and omitted failures | High; systematically inflates accuracy and hides operational failures |
| 4 | Missing three-attempt retry/validation protocol | High; changes which transient/format failures become predictions |
| 5 | Cross-question memory within each case plus unsorted glob order | Potentially high; can leak prior answers/context and introduces order dependence |
| 6 | Current HF dataset schema is incompatible with notebook input schema | High operational impact; notebook cannot run as checked out in this workspace |
| 7 | Different bound tool set in the local eight-tool runner | Medium/high; controller sees different action space |
| 8 | T4 CPU/offload runtime versus RTX 6000 benchmark configuration | Medium; may change numerical behavior through dtype/quantization and substantially changes timing |
| 9 | Answer parser differs from paper's unspecified regex | Medium; changes validity/correctness of verbose responses |
| 10 | Tool exceptions abort instead of becoming observations | Medium; affects samples with tool runtime failures |
| 11 | No explicit `tmax`/timeout | Low/medium unless loops or calls hang |
| 12 | Mutable controller/model revisions and broad dependency ranges | Medium for long-term reproducibility |
| 13 | Sequential execution of batched tool calls | Low for accuracy, high for timing |

## Approved minimal fixes applied

The following user-approved paper-fidelity changes were applied:

1. **Remove only the case-ID resume filter** from `benchmark_medrax.ipynb`.
2. **Implement a three-attempt per-question loop** covering controller/API exceptions, timeouts, malformed responses, and invalid answer extraction. Keep the prompts, models, tools, and LangGraph architecture unchanged.
3. **Use one documented A-F extraction function** for the MedRAX runner and analyzer. Because the paper does not disclose its exact regex, the shared rule extracts the first standalone A-F token, case-insensitively.
4. **Log exactly one terminal record per question ID.** After the third failed attempt, write `status="error"`, preserve `correct_answer` and metadata, and set `model_answer` to null so aggregation counts it as incorrect.
5. **Make aggregation manifest-backed.** Use the existing positional `benchmark_dir` argument (currently ignored) or the released metadata JSONL to enumerate all 2,500 expected question IDs. Join predictions by question ID, reject duplicates, and count missing/invalid/error predictions as incorrect. Assert an expected count of 2,500 for ChestAgentBench.
6. **Disable `--max-questions` for paper mode** or fail if it is supplied with the MedRAX paper protocol.

### Intentionally unchanged

- Cross-question memory remains one thread per contiguous case, matching the notebook.
- Benchmark ordering remains unchanged: JSONL line order for the released format and the existing iteration order for the legacy format.
- Prompts, controller model, temperature, top-p, six-tool selection, LangGraph workflow, and reasoning strategy are unchanged.

Schema compatibility between `benchmark/questions/` and `chestagentbench/metadata.jsonl` should be solved in the runner without changing question content. A compatibility adapter may resolve local image paths at runtime, but it must preserve the paper's exact question text, image set, prompt, and ordering.

## Fixes applied in this audit

- Removed the `case_id <= 17158` resume checkpoint from the paper-facing notebook.
- Added the released JSONL runtime adapter while retaining legacy question-tree support.
- Added one shared standalone A-F regex parser for the MedRAX runner and analyzer.
- Added three attempts for exceptions, timeouts raised by dependencies, malformed responses, and invalid answer choices.
- Added exactly one terminal log record after success or the third failed attempt.
- Added manifest-backed MedRAX aggregation with 2,500 unique IDs; missing, invalid, error, and timeout outcomes count as incorrect.
- Disabled `--max-questions` for the MedRAX paper protocol.
- Added this documentation report.
- No experiment was executed.

Pre-existing compatibility changes in the worktree (`main.py`, `medrax/tools/generation.py`) are outside this audit. They are not imported by the paper-facing notebook's metric aggregation, although runtime placement in `main.py` affects custom local runners.

## Remaining limitations even after the proposed code fixes

1. The exact GPT-4o snapshot used for the paper is not pinned and the alias is mutable.
2. The exact answer regex is not disclosed in the paper and repository implementations conflict.
3. Per-case memory reuse versus per-question isolation is not specified by the paper.
4. Published raw predictions, retry logs, and the exact 63.1% numerator are absent.
5. Model checkpoint revisions are generally unpinned.
6. Most dependency versions are lower-bounded rather than locked; only the Transformers Git revision is pinned.
7. The current checkout is a fork remote rather than the canonical paper repository.
8. Hardware and runtime placement differ from the RTX 6000 setup.
9. Temperature 0.2 introduces stochasticity and no controller seed is recorded.
10. The released compact metadata has one sample without an image reference; the paper protocol requires it to remain in the denominator.

## Expected reproducibility

The approved code path now reproduces the stated ChestAgentBench evaluation protocol with a fixed 2,500-question denominator. Numerical reproduction of the published 63.1% result is not expected with the current local Gemini/T4 runtime. Exact numerical reproduction will still require the original GPT-4o snapshot/access path, six-tool benchmark configuration, equivalent model checkpoints/runtime, and clarification of cross-question memory behavior.

## Final verification answers (current repository)

**Are all 2,500 questions evaluated?** Yes. The runner consumes all 2,500 released JSONL records in order and refuses to start if the manifest count differs.

**Are failed samples counted as incorrect?** Yes. Missing, invalid, error, and timeout terminal outcomes receive no correct credit and remain in the manifest-backed denominator.

**Is the denominator always 2,500?** Yes for `--model medrax`; the analyzer validates a 2,500-question manifest and rejects optional truncation.

**Is the implementation faithful to the paper?** It now matches the paper's stated ChestAgentBench protocol for benchmark size, retries, terminal outcomes, answer extraction, and denominator. Exact numerical reproduction remains limited by the unavailable original GPT-4o snapshot, current Gemini/T4 runtime, unresolved per-case-memory ambiguity, and unpinned model revisions.

## Primary references

- Paper: https://arxiv.org/abs/2502.02673
- Canonical repository: https://github.com/bowang-lab/MedRAX
- Paper-facing runner: `experiments/benchmark_medrax.ipynb`
- Metric aggregation: `experiments/analyze_axes.py`
- Released compact benchmark: `chestagentbench/metadata.jsonl`
