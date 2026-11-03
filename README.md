# MFDE

MFDE implements dynamic ensembling of frozen pathology foundation-model features for whole-slide image classification under a single-source domain-generalization setting.

## What this code implements

- Load aligned, variable-length patch-feature bags from multiple frozen encoders and read explicit patient-disjoint train, validation, and test partitions.
- Project encoder-specific features into a common space and adapt them with prototype-assisted shared remapping.
- Pool instances with multi-head MIL attention and compute slide-dependent branch weights from bag features and attention statistics.
- Fuse branch representations through weighted aggregation and cross-model attention to predict slide-level labels.
- Train with main and auxiliary classification, probability-based routing supervision, and multi-stage cosine feature preservation using an EMA teacher.
- Select checkpoints by source-domain validation AUC and evaluate the selected checkpoint on the target domain.
- Optionally record a separate Oracle-selected checkpoint and its target metrics along the same source-trained trajectory.
- Accumulate variable-length slide bags, update the EMA teacher after each optimizer step, and export predictions, branch weights, and optional per-branch attention.
- Generate one-factor-at-a-time sensitivity configurations, collect run results, and plot mean AUC with sample standard deviations.

Inputs are frozen patch-feature tensors and a split CSV. Outputs include a source-validation-selected checkpoint, training history, evaluation metrics, configuration records, and sensitivity plots when measured run results are supplied. Feature extraction is performed separately; the foundation encoders are not trained by this package.

## Release status and TODO

The organized MFDE network and the current training, evaluation, and sensitivity utilities are included here. Public release of the complete reproducibility package is in preparation. Complete code and pretrained models will be released after publication.

- [ ] Publish the MFDE pipeline with consistent concept-based module names.
- [ ] Publish the complete training and evaluation package and final pool-selection utilities.
- [ ] Publish pretrained model/checkpoint specifications, trained MFDE checkpoints, and reproduction instructions.
- [ ] Complete and publish the remaining parameter settings, strategy details, and experiment configurations.
- [ ] Publish the final split/protocol manifests and evaluation instructions.

## Evaluation and model-pool protocol

Source-only training produces candidate checkpoints. ID selects by source-domain validation AUC; Oracle selects by target-domain AUC for diagnostic comparison. Both report the selected checkpoint's target-domain AUC. Encoder-pool selection uses source-domain validation information exclusively, and the selected encoder list is fixed before target evaluation. Oracle results are not fed back into pool selection. Patient identities remain disjoint across training, validation, and test partitions, including across scanners and multiple slides.

The `train.py` entry point delegates training and evaluation to `mfde/multi_engine.py`. ID selection is always recorded. Optional Oracle reporting stores a separate checkpoint and evaluation record; source-validation AUC continues to govern ID selection and optional early stopping. Encoder membership is configured before a run. Automated pool search is part of the release TODOs above.

## Modules

| Component | Implementation |
| --- | --- |
| MFDE | mfde/network.py: MFDE |
| Prototype-assisted shared remapping | mfde/components.py: PrototypeAssistedSharedRemapping |
| MIL pooling | mfde/components.py: MultiHeadMILAttention |
| Dynamic routing | mfde/components.py: WeightAssignmentOperator |
| Cross-model fusion | MFDE.cross_model_attention and MFDE.fusion_ffn |
| EMA prior preservation | mfde/prior_preservation.py |
| Training objective | mfde/objective.py |
| Training, validation, checkpoint selection, and prediction export | mfde/multi_engine.py: MFDEEngine |

## Installation and input

Use Python 3.10 or later and install `pip install -r requirements.txt` in your own environment. Select a PyTorch build appropriate for your GPU. No weights, datasets, credentials or personal paths are distributed.

Supply frozen features as `[patches, channels]` float tensors saved in `<encoder_directory>/<ID>.pt` or `<encoder_directory>/pt_files/<ID>.pt`. All encoders must represent the same ordered patch set. Equal patch counts are checked; identical patch identities and ordering must also be established during feature extraction. No truncation, padding or random patch subsampling is performed.

The split CSV must contain `Case,ID,Label,DomainLabel,Split`. Label must be an integer from 0 to `num_classes-1`. Split values are train, val and test. Case identifiers must identify actual patients, not merely slides. Train and val must share one source domain; test domains must be disjoint from it. Empty validation, duplicate slide IDs, missing features and case overlap are rejected. The program does not invent a replacement split. Record a case mapping if patient identities differ across scanners or cohorts.

Create a local JSON configuration from configs/base.json and set environment variables MFDE_SPLIT_CSV, MFDE_UNI_FEATURES, MFDE_GIGAPATH_FEATURES and MFDE_PHIKON_FEATURES. Use distinct run output directories. PANDA has six output classes in the supplied example; use two for Camelyon and HER2. Set the common feature dimension and objective coefficients in the configuration file. Private paths belong in environment variables or ignored local configurations; scientifically necessary hyperparameters remain visible.

## Run

```text
python train.py --config configs/local_panda.json
python train.py --config configs/local_panda.json --oracle-reporting
python experiments/make_sweep.py --config configs/local_panda.json --output experiments/panda_sweep
python experiments/run_sweep.py --manifest experiments/panda_sweep/manifest.csv --results experiments/panda_results.csv
python experiments/plot_sensitivity.py --input experiments/all_results.csv --output figures/sensitivity
```

The sweep runner prints commands by default. Add `--execute` to run them. It never starts remote jobs or uploads results. Use distinct output directories for separate runs. In the default ID mode, the target split is evaluated after choosing the best source-validation checkpoint. With `--oracle-reporting`, target AUC is additionally recorded each epoch for the separate Oracle checkpoint. AUC is binary or macro one-vs-rest multiclass; undefined AUC raises an error rather than becoming zero. Each output contains a split hash, a path-free protocol hash, the exact configuration, validation history and selected target AUC. Local run logs contain the local paths supplied by the user and should not be published without review.

## Training engine

`MFDEEngine` provides `train`, `validate`, `testing`, and `learning`. It accepts the package's `(features, labels)` batches and legacy `(case_id, slide_id, features, labels, coordinates, domain)` batches. Feature dictionaries keep the encoder names used by the network. Evaluation loaders should preserve slide order; the standard entry point uses sequential validation and test loaders.

Training accumulates the configured number of slide bags before a single optimizer step. Losses are averaged over the actual bags in each accumulation group, including the final partial group. The EMA teacher is updated once after each optimizer step. Main classification, branch auxiliary classification, detached-probability routing supervision, and the configured prior-preservation objective are calculated in `objective.py`. The effective regularization coefficient is applied once. Optional logit normalization affects the main training loss and uses the configured temperature.

The `engine` configuration fields are:

| Field | Default | Function |
| --- | --- | --- |
| `oracle_reporting` | `false` | Record target-selected diagnostic checkpoints alongside ID selection. |
| `early_stopping_patience` | `null` | Train for all epochs; a positive integer enables stopping after that many source-validation epochs without improvement. |
| `parameter_averaging_start_update` | `null` | A positive optimizer-update index enables arithmetic parameter averaging from that update onward. |
| `binary_threshold` | `0.5` | Fixed probability threshold for binary precision, recall, F1, and accuracy. |
| `export_predictions` | `true` | Export labels, probabilities, bag features, and routing weights for the selected checkpoint. |
| `export_attention` | `false` | Also export per-slide attention, retaining separate encoder and attention-head axes. |

Parameter averaging uses PyTorch `AveragedModel`, following the averaging operation in the supplied engine. Its configured start is counted in optimizer updates. Evaluation and checkpoint saving use the same averaged weights after averaging starts. This option implements arithmetic trajectory averaging; a full validation-valley SWAD policy is a separate release item.

Metrics include ROC-AUC, average precision, accuracy, precision, recall, F1, and binary PPV/NPV. Multiclass AUC and average precision use macro one-vs-rest aggregation. Binary threshold-dependent metrics use the configured fixed threshold; the target labels are not used to optimize that threshold. AUC remains threshold-independent.

Each run writes:

- `best_source_validation.pt`: tensor-only model weights selected by source-validation AUC.
- `best_oracle.pt`: separate target-selected weights when Oracle reporting is enabled.
- `history.json`: per-epoch loss components, validation metrics, training time, update count, and optional target AUC.
- `engine_result.json`: full ID metrics and optional separately labeled Oracle metrics and selected epochs.
- `result.json`: the existing sensitivity-run summary, with optional `oracle_target_auc` and `oracle_selected_epoch` fields.
- `target_id_outputs.pt` and optional `target_oracle_outputs.pt`: exported outputs when prediction export is enabled.

Exported instance weights have shape `[encoders, patches]` per slide; raw attention-head scores have shape `[encoders, heads, patches]`. Routing weights are stored separately. Legacy batches also retain supplied patch coordinates. These tensor records can be loaded using `torch.load(path, weights_only=True)`. Validation and test collection preserve variable patch counts across slides.

The supplied engine's external `cal_reg_loss` and `MCCHandler` dependencies remain release TODOs. The package uses the explicitly identified `manuscript_reference` preservation implementation. Additional objectives can be supplied through the engine's `objective` callable interface; their configuration and provenance should be recorded alongside run results.

The three sweep axes are prototype count K = 8, 16, 32, 64, 128; routing-target temperature tau = 0.025, 0.05, 0.1, 0.2, 0.4; and effective prior-preservation coefficient lambda = 0, 0.25, 0.5, 1, 2. Other settings are held fixed. The zero coefficient is an ablation endpoint. Baseline settings are K=32, tau=0.1 and lambda=1. The complete design has 13 unique configurations per source task and seed because the baseline is shared by all axes. The generator creates three seeds (42, 43, 44) for NEW runs; replace these consistently if matching historical seeds is required.

The author-supplied aggregate sensitivity results and the corresponding 1-by-3 figure are included in `experiments/reported_results/`. The CSV contains dataset, parameter axis, parameter value, Oracle-selected target mean AUC, and ID-selected target mean AUC. These aggregate records support analysis of K = 32, tau = 0.1, and effective lambda = 1. The regularization-zero records agree with the component-ablation results. The run-level plotting utility above accepts seed/task records and computes its own aggregation; the supplied aggregate CSV has a separate, documented schema. Read experiments/EXPERIMENTS.md for the run-level workflow.

## Verification

Run `python -m unittest discover -s tests -v`. Synthetic tests cover forward gradients, effective coefficient scaling, partial accumulation groups, EMA updates, independent ID/Oracle selection, source-based early stopping, averaged checkpoint consistency, attention export, multiclass metrics, and the complete training entry point. Implementation provenance, the reference objective, and checkpoint-key conversion are documented in PROVENANCE.md.
