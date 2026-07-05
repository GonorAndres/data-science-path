# Exploratory Notebooks

This folder is a sandbox. It is not part of the portfolio surface area.

## What lives here

- Throwaway EDA on new datasets before a project exists.
- Technique spikes: a single technique tested against a toy dataset to decide if it is worth promoting into `techniques/`.
- Scratch work for interview prep or live coding practice.

## What does not live here

- Anything that belongs to a shipped project. That lives under `projects/<name>/notebooks/`.
- Anything that belongs to a technique reference. That lives under `techniques/<name>/`.
- Anything with customer or private data. This folder is gitignored-adjacent: commit only notebooks that can be public.

## Naming convention

```
<YYYY-MM-DD>_<short-topic>.ipynb
```

Example: `2026-05-02_uplift_bulk_experiment.ipynb`

## Lifecycle

A notebook in this folder either:

- Gets promoted: code moves into `scripts/utils/` or a project, the notebook moves out.
- Gets archived: rename to `_archive/<filename>.ipynb` once no longer in active use.
- Gets deleted: if it never went anywhere, remove it so the folder stays readable.

The expectation is this folder has fewer than ten live notebooks at any time.
