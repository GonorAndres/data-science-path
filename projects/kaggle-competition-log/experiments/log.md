# Experiment Log (Template)

Keep one table per competition. Add a row for every experiment.

| Date | Experiment | Hypothesis | Change | CV score | Public LB | Notes |
|------|------------|------------|--------|----------|-----------|-------|
| YYYY-MM-DD | baseline | establish floor | lightgbm default | X.XXX | X.XXX | initial submission |
| YYYY-MM-DD | feat_v1 | domain features help | add N features | X.XXX | X.XXX | ... |
| ... | ... | ... | ... | ... | ... | ... |

## Instructions

- Log EVERY experiment, even failures.
- Write the hypothesis BEFORE running.
- Record CV score and public LB side by side.
- If CV helps but LB hurts (or vice versa), flag for investigation.
- At end of competition, the log is the backbone of `writeup.md`.
