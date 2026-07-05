# Competition Strategy

## Order of operations

1. **Read every word of the overview, data description, and evaluation metric.** The metric is often the single most important piece; different metrics favor different models.
2. **Study past winners.** The Kaggle forum and writeups for similar past competitions often reveal 80 percent of what works.
3. **Baseline fast.** First submission on day one, even if it is a trivial model.
4. **Profile the data.** Label distribution, missingness, correlations, time structure.
5. **Iterate on features.** Usually the biggest lever.
6. **Iterate on models.** After features stabilize.
7. **Ensemble last.** Combining multiple strong models gives the final lift.
8. **Writeup.** During the competition, keep a running log.

## What matters

- **Cross-validation.** Covered in `02_cv_strategy_for_competitions.md`.
- **Leakage.** Covered in `04_leakage_checklist.md`.
- **Feature engineering.** Domain-specific; learn the domain.
- **Ensembling.** See `03_ensembling_basics.md`.
- **Compute discipline.** Track what you can reproduce; avoid notebooks that work only on your local machine.

## What does not matter (as much as people think)

- Exotic architectures. Simple models + strong features usually beat fancy models.
- Hyperparameter tuning beyond a modest grid. Return on tuning is usually lower than return on features or ensembling.
- Public notebooks with huge upvote counts. Often over-fit to public LB.

## Public LB vs private LB

Kaggle splits the test set:

- Public LB (visible during competition): 20-30 percent of test.
- Private LB (revealed at end): 70-80 percent.

Relying on public LB for model selection overfits to it. Validate with robust CV first; treat public LB as sanity check.

## Shake-ups

Some competitions have large differences between public and private LB rankings. Known "shake-ups" happen when:

- Public LB has heavy label noise.
- Public LB is a biased subset (e.g., one country only).
- Top public LB solutions are over-specialized.

Mitigate by trusting robust CV over public LB.

## Cross-links

- `02_cv_strategy_for_competitions.md`
- `03_ensembling_basics.md`
- `04_leakage_checklist.md`
