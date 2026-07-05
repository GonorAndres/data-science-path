# Notebook 04: Error Analysis and Responsible AI (outline)

## Sections

### 1. Per-language fairness check

Macro F1 per language. If gap > 3 pp, investigate.

### 2. Error taxonomy

For each language, sample 50 misclassifications. Tag each with one of:

- Sarcasm
- Negation
- Mixed sentiment
- Label noise (debatable)
- Code-switching
- Very short
- Typos / slang
- Domain-specific term

Produce a table of counts.

### 3. Confidence-vs-correctness

Reliability diagram. Is a 0.9 prediction 90 percent correct?

### 4. Category-level breakdown

If `product_category` is available in the dataset, per-category F1 bar chart.

### 5. Stress tests

Run the hand-crafted stress tests from `tests/test_predict.py`. Document outcomes.

### 6. Action items for v2

Concrete list: "Add 500 sarcastic examples to training", "Use back-translation augmentation for Spanish", "Try XLM-R base".
