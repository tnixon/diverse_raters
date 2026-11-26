# diverse_raters

Can a diverse ensemble of raters keep a language-model optimizer from reward hacking?

## The claim

**At matched rater accuracy, optimizing against a diverse, randomly sampled ensemble of raters produces less reward hacking than optimizing against a single rater.**

The construct is toxicity, framed as detoxification: rewrite a comment so it is no longer toxic. A *hack* is a rewrite that the training signal scores as clean but that is not.

## Design

- **Raters:** small DeBERTa-size encoders trained on Jigsaw Civil Comments, using the individual annotations. Measuring Hate Speech is used only for a cross-dataset check.
- **Optimizer:** search. An LLM proposes rewrites, and the top ones under each condition's signal are kept, with a fixed and equal budget per condition. Every candidate is logged with every rater's score.

### Conditions

| # | Training signal |
|---|-----------------|
| 1 | Single rater |
| 2 | Single rater plus independent noise |
| 3 | Random-partition raters, one sampled per episode |
| 4a | Cluster-partition raters |
| 4b | Annotator-partition raters |
| 5 | Raters on different base models; m of n sampled, mean minus a disagreement penalty |

### Measurements

- **Matched competence:** accuracy on a validation set carved out before any partitioning.
- **Hack gap:** gain in training reward minus gain in trusted score.
- **Transfer matrix:** how rewrites optimized against one rater score under the others.
- **Cross-dataset check:** the same measurements on Measuring Hate Speech.
- **Trusted score:** held-out annotators, an LLM judge from a different model family, and a human sample. Each is reported separately.

### Pre-registration

Predictions are committed to [PREDICTIONS.md](PREDICTIONS.md) before any ensemble run and frozen after the first one.

## Status

Work in progress, in a project sprint ending mid-December 2026. The work is built in four reviewed stages: (1) skeleton and workspace wiring, (2) data, (3) raters, (4) search and evaluation. Stages 1 and 2 exist so far (stage 2 is awaiting review), so there is no rater training, search, or evaluation code yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Databricks workspace with Unity Catalog.

```bash
uv sync
uv run ruff check .
uv run pytest
```

Credentials come from the environment, or from the Databricks SDK's default authentication (for example a CLI profile). Inside Databricks they are picked up automatically.

```bash
export DATABRICKS_HOST=...            # workspace URL
export DATABRICKS_TOKEN=...           # personal access token
# or: export DATABRICKS_CONFIG_PROFILE=<profile>
```

Then run `notebooks/00_setup.py` in the workspace (for example from a Git folder). It creates, if missing, the catalog `diverse_raters`, the schemas `raw`, `curated`, `results`, and `models`, the volume `raw.downloads`, and the MLflow experiment `/Shared/diverse-raters`. It finishes with a smoke-test run that registers `diverse_raters.models.smoke_test`.

## Data

Data is never committed. It is downloaded into the `diverse_raters.raw.downloads` volume.

- **Jigsaw Civil Comments, individual annotations:** Kaggle. You must accept the competition rules with your Kaggle account before downloading, and store a Kaggle API token (from kaggle.com/settings/api) in the Databricks secret scope named in `conf/data.yaml` (default scope `kaggle`, key `api_token`).
- **Measuring Hate Speech:** used only for the cross-dataset check. Downloaded from Hugging Face at the revision pinned in `conf/data.yaml`.

Two notebooks, run in order, carve out the up-front splits before anything else touches the data.

1. `notebooks/01_ingest.py` downloads both datasets into the volume, parses them into Delta tables in `raw`, and logs a `data_profile` MLflow run: labels per worker, the label share each candidate held-out fraction would remove, and the count of exact duplicate texts. No split exists yet.
2. Read that profile, then set `heldout_annotator_fraction` and `min_heldout_annotations` in `conf/data.yaml` (they ship as `null`, and the next notebook refuses to run without them). Labels per worker are heavily skewed, so a fixed 10% of workers could be a small or a large share of the labels.
3. `notebooks/02_splits.py` builds the splits and curated tables and logs a `data_prep` run.

How the splits work:

- **Comments** go to `train` / `validation` / `search` by a stable hash of their *normalized text*, not their id, so identical text under different ids always lands in the same split. Near-duplicates are not caught.
- **Held-out annotators** are chosen by a stable hash of the worker id. Their labels go only to `curated.jigsaw_heldout_annotators`, on every comment split. They never see the rewrites, so they cannot score them directly. Instead their labels train a held-out rater model, which scores rewrites as one of the trusted sources. That is why `min_heldout_annotations` exists: the held-out set must be large enough to train a competent rater.
- **Labels are rebuilt, not copied.** The `toxicity` column in Kaggle's `all_data.csv` aggregates every annotator, held-out ones included, so it is never read. Each curated table carries `toxicity_mean` and `annotator_count` computed from only the annotators that table may contain.
- **Search source comments** (`curated.jigsaw_search_sources`) are drawn from the search split: at least `min_annotators` training annotators, `toxicity_mean` at or above the threshold, one id per distinct text, ordered by a seeded hash and truncated to `count` (200 by default).
- The split tables carry a `source` column (currently `jigsaw`), so another dataset can be added later as new rows.

Each curated table, plus Measuring Hate Speech, is logged as an MLflow dataset pinned to its Delta version.

## Layout

```
src/dre/       library code (settings, data stage)
conf/          experiment configs
notebooks/     thin Databricks notebooks
tests/         unit tests (no network, no paid API calls)
AGENTS.md      invariants and rules for contributors and coding agents
PREDICTIONS.md pre-registered predictions
```

## Open design decisions

- Whether the candidate pool is shared across conditions or built per condition.
- Where the annotator split comes from. For now it is a hash of the worker id, with the fraction set from the annotator profile; stratifying by annotation volume is possible if the profile shows a few heavy workers dominate.
- Whether the search source comments should also be spread across clusters. Clusters do not exist until stage 3, so this would be a re-selection with a new table version.
- Whether to add the Specialized Rater Pools dataset. Left out for now; the split tables can take it as a new `source`.
- Where GPU training runs.

## Licence

MIT. See [LICENSE](LICENSE).
