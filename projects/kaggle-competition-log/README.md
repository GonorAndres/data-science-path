# kaggle-competition-log

## Resumen (ES)

**Problema:** documentar un intento serio en una competencia viva de Kaggle, desde la primera exploracion hasta el ultimo submission. El objetivo es la disciplina del ciclo de iteracion, no la medalla.

**Metodo:** una competencia a la vez; notebooks de EDA, baseline, feature engineering, modelos, ensembling; log diario de experimentos; un writeup final con lecciones aprendidas.

**Resultado objetivo:** un submission valido, un ranking publico documentado, y un writeup que un recruiter pueda leer en 10 minutos y entender el proceso.

## Summary (EN)

**Problem:** document a serious attempt at a live Kaggle competition, from first exploration to final submission. The goal is iteration discipline, not the medal.

**Method:** one competition at a time; notebooks for EDA, baseline, feature engineering, models, ensembling; daily experiment log; a final writeup with lessons learned.

**Target result:** a valid submission, a documented public ranking, and a writeup a recruiter can read in 10 minutes and understand the process.

---

## Why this project

Kaggle ranking signals rigor, speed, and comfort with external ML communities. Even a non-medal finish shows:

- You know how to set up a reproducible pipeline.
- You iterated, not just submitted once.
- You learned something specific enough to write about.

Many Mexican DS teams ask about Kaggle or similar (DrivenData, AIcrowd) in interviews.

## Structure

```
kaggle-competition-log/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_competition_strategy.md
│   ├── 02_cv_strategy_for_competitions.md
│   ├── 03_ensembling_basics.md
│   └── 04_leakage_checklist.md
├── notebooks/
│   ├── 00_setup.md
│   ├── 01_eda.md
│   ├── 02_baseline.md
│   ├── 03_feature_engineering.md
│   ├── 04_models.md
│   └── 05_ensemble_and_final.md
├── experiments/
│   └── log.md                 # daily what-I-tried / what-worked
└── writeup.md                 # final narrative
```

## Status

This project scaffolds the structure; a real competition log is added when one is active. To start:

1. Pick a live Kaggle competition. Data scientists usually join one at a time.
2. Copy the notebooks outline and start filling `00_setup.md` with the competition-specific info.
3. Log every experiment (what you changed, what the CV score was, what the public score was).
4. Submit frequently; early baselines beat late-over-engineered ones.
5. After deadline, write the retrospective.

## References

- `docs/theory/03_cross_validation_strategies.md`.
- `docs/theory/04_data_leakage_patterns.md`.
- Anokhin & Lopuhin. "Winning the Kaggle Competition". Various interviews.
- Kaggle Winning Solutions writeups (kaggle.com/code, tag: winning-solution).
