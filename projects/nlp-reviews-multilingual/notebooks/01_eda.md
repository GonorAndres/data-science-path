# Notebook 01: Exploratory Data Analysis (outline)

Convert to `.ipynb` when running the project.

## Sections

### 1. Load the data

```python
from src.data import load_split, summarize
print(summarize())
```

Expected: train/val/test counts per language, class distribution per language.

### 2. Class balance

Bar chart: share of negative/neutral/positive per language. Confirm moderate imbalance (~20 neg, ~30 neu, ~50 pos).

### 3. Review length distribution

Histogram of word count and token count per language. Identify the 95th percentile. Confirm the 128-token cap does not truncate the bulk of data.

### 4. Top words per class (quick inspection)

Wordcloud or top-N bar chart per class per language. Sanity check: "malo", "pesimo", "horrible" show up in negatives. "Excelente", "recomendado" in positives.

### 5. Sanity check for label noise

Sample 20 random reviews per class, read. Note percent where you would disagree with the automated star-to-class mapping. Typical finding: ~10 to 15 percent of 3-star reviews feel positive or negative to a reader.

### 6. Cross-language duplicate check

Count rows where the same text appears more than once. Report. These should be deduplicated before splitting.

### 7. Takeaways

- Class distribution.
- Length summary.
- Label noise estimate.
- Dataset issues to address in preprocessing.
