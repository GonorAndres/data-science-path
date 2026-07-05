# Notebook 02: TF-IDF + LogReg Baseline (outline)

Convert to `.ipynb` when running the project.

## Sections

### 1. Recap of EDA findings

One-paragraph summary from notebook 01.

### 2. Pipeline definition

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
```

Explain each hyperparameter choice.

### 3. Train the baseline

Run `src.baseline.train_baseline()` or inline the same pipeline.

### 4. Evaluation

- Confusion matrix on validation.
- Per-class precision, recall, F1.
- Macro F1 per language.

### 5. Coefficient inspection

For each class, show the top 20 most positively and negatively associated tokens per language. Narrate what the model learned.

### 6. Error analysis

Sample 20 misclassified reviews per language. Categorize errors (sarcasm, negation, mixed, code-switch, typos).

### 7. Takeaways

- Baseline macro F1 per language.
- Error types to target with the transformer.
- What to expect as the floor for the transformer comparison.
