# Feature Engineering

## What it is

Feature engineering is the process of transforming raw data into meaningful inputs that a machine learning model can actually learn from. Think of it as translating the real world into a language your model understands. Raw data -- dates, categories, free text, raw measurements -- rarely arrives in a form that algorithms can directly exploit. Feature engineering bridges that gap by creating, combining, transforming, and encoding variables so that the signal in your data becomes visible to the model. It is widely considered the single most impactful step in a machine learning pipeline: a mediocre algorithm with great features will almost always beat a fancy algorithm with poor features.

## Why it matters

For actuarial and data science practitioners, feature engineering is where domain knowledge meets technical skill. In insurance pricing, for example, the difference between a raw "date of birth" column and a well-constructed "driver age band" feature can be the difference between a model that prices risk accurately and one that misses key patterns. Actuaries have been doing feature engineering for decades -- they just called it "rating factor development" or "exposure analysis." Understanding modern feature engineering techniques lets you bring that actuarial intuition into machine learning workflows while also leveraging data-driven transformations that go beyond traditional approaches.

Feature engineering also directly affects model interpretability. Well-engineered features often have clear business meanings, making it easier to explain your model to underwriters, regulators, and business stakeholders -- a critical requirement in insurance and financial services.

## When to use it

- **Before training any supervised learning model** -- feature engineering is almost always part of the pipeline.
- **When your raw data contains categorical variables** that need encoding for algorithms that expect numeric input.
- **When you have date/time columns** that contain latent information (day of week, seasonality, tenure).
- **When domain knowledge suggests interactions** between variables (e.g., age combined with vehicle type in motor insurance).
- **When you want to improve model performance** without changing the algorithm -- feature engineering is often the highest-leverage intervention.
- **When working with high-cardinality categorical variables** (e.g., ZIP codes, occupation codes) where naive encoding would create thousands of sparse columns.
- **When building actuarial pricing or reserving models** where rating factors need to be constructed from raw policy and claims data.

## How it works (intuitive)

### The big picture

Imagine you are trying to predict how much a restaurant meal will cost. The raw data might include: restaurant name, date of visit, number of guests, items ordered (as text), and location. None of these are directly usable by a regression model in their raw form. Feature engineering turns them into things like: "is it a weekend?", "number of courses ordered", "average price tier of the neighborhood", "is it a holiday?", and so on. Each transformation extracts a piece of information that was hidden in the raw data.

Feature engineering generally falls into several categories:

### 1. Encoding categorical variables

Categorical variables (like "vehicle type" or "province") are labels, not numbers. Models need numbers. The question is *how* to convert them.

**One-hot encoding** creates a separate binary (0/1) column for each category. If you have a "color" column with values Red, Blue, Green, you get three new columns: `color_Red`, `color_Blue`, `color_Green`. Each row has a 1 in exactly one of these columns.

- *Analogy*: It is like giving each category its own light switch. Only one switch is on at a time.
- *When to use*: When the number of categories is small (say, under 20-30) and there is no natural ordering.
- *Watch out*: With many categories, you get an explosion of columns (the "curse of dimensionality"). A ZIP code column with 10,000 unique values would create 10,000 new columns.

**Ordinal encoding** assigns integers based on a meaningful order. "Low", "Medium", "High" becomes 1, 2, 3. This preserves the ranking information.

- *Analogy*: It is like assigning grades -- A=4, B=3, C=2, D=1. The numbers reflect a real ordering.
- *When to use*: When categories have a natural order (education level, policy tier, credit rating band).
- *Watch out*: Do not use ordinal encoding when there is no real order. Encoding "Red=1, Blue=2, Green=3" implies Blue is "between" Red and Green, which is nonsensical.

**Target encoding** (also called mean encoding) replaces each category with the mean of the target variable for that category. If customers in Province A have an average claim rate of 0.12, then every row with Province A gets the value 0.12.

- *Analogy*: It is like replacing each school's name with its average test score. The name disappears, but the performance information stays.
- *When to use*: High-cardinality categoricals (ZIP codes, occupation codes) where one-hot would create too many columns.
- *Watch out*: Target encoding leaks information from the target into the features, which can cause overfitting. Always use it with cross-validation or smoothing (adding a regularization term that blends the category mean with the global mean).

### 2. Binning (discretization)

Binning converts continuous variables into categorical ones by grouping values into ranges. Age might become "18-25", "26-35", "36-50", "51-65", "65+".

- *Analogy*: It is like sorting mail into bins by ZIP code prefix rather than exact address.
- *Why do it*: It can capture non-linear relationships (the effect of age on claim frequency might jump at certain thresholds), reduce noise, and improve interpretability. Actuaries have been doing this forever with rating tables.
- *Methods*: Equal-width bins (same range per bin), equal-frequency bins (same number of observations per bin), or domain-driven bins (based on business knowledge, like actuarial age bands).

### 3. Interaction features

Sometimes the effect of one variable depends on another. A young driver in a sports car is riskier than what you would predict from "young" and "sports car" separately. An interaction feature captures this by combining two (or more) variables.

- *Simple approach*: Multiply two features together. `age * engine_size` creates a new feature that is large when both age and engine size are large.
- *Categorical interactions*: Combine two categorical variables into one. `gender_region` might have values like "M_Urban", "F_Rural", etc.
- *Analogy*: It is like recognizing that "rain + rush hour" creates a traffic effect that is worse than either alone.

### 4. Date and time features

Raw dates are almost never useful on their own. But they contain a wealth of information:

- **Temporal components**: Year, month, day of week, hour, quarter, is_weekend, is_holiday.
- **Cyclical encoding**: Months and hours are cyclical (December is close to January). Encode them using sine and cosine transforms: `month_sin = sin(2 * pi * month / 12)`, `month_cos = cos(2 * pi * month / 12)`.
- **Time since event**: Days since last claim, months since policy inception, years of tenure.
- **Seasonality flags**: Is this in hurricane season? Is this during winter driving conditions?

### 5. Domain-specific features for insurance and actuarial work

This is where actuarial training provides a real edge:

- **Exposure**: The fraction of a policy year that a policy was in force. A policy active for 6 months has exposure = 0.5. Almost every frequency model needs this as an offset.
- **Loss ratios**: Incurred losses divided by earned premium -- a core actuarial metric that can be a feature in models predicting future profitability.
- **Claims development features**: For reserving models, features like "months since accident", "number of payments to date", "ratio of paid to incurred" capture where a claim is in its lifecycle.
- **Territory features**: Aggregated statistics by geographic region -- population density, average income, historical claim frequency.
- **Vehicle/property features**: Age of vehicle, replacement cost, construction type, protection class.
- **Policyholder history**: Number of prior claims, years claim-free, policy tenure, number of coverage changes.

## Key concepts

### One-hot encoding

For a categorical variable $X$ with $k$ categories, one-hot encoding creates $k$ binary variables:

$$X_i^{(j)} = \begin{cases} 1 & \text{if observation } i \text{ belongs to category } j \\ 0 & \text{otherwise} \end{cases}$$

In linear models, you typically drop one category (the "reference" category) to avoid perfect multicollinearity, giving you $k-1$ columns. Tree-based models do not need this.

### Target encoding with smoothing

For category $c$ with $n_c$ observations and mean target value $\bar{y}_c$, the smoothed target encoding is:

$$\text{encoded}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$

Where $m$ is a smoothing parameter. When $n_c$ is large, the encoding is close to the category mean. When $n_c$ is small, it shrinks toward the global mean. This is essentially credibility weighting -- a concept actuaries know well from credibility theory.

### Cyclical encoding

For a feature with period $P$ (e.g., $P=12$ for months, $P=7$ for day of week):

$$x_{\sin} = \sin\left(\frac{2\pi \cdot x}{P}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot x}{P}\right)$$

This ensures that the model understands that month 12 (December) is close to month 1 (January).

### Polynomial and interaction terms

For features $x_1$ and $x_2$, common derived features include:

- **Interaction**: $x_1 \cdot x_2$
- **Polynomial**: $x_1^2$, $x_1^3$
- **Ratio**: $x_1 / x_2$ (when $x_2 \neq 0$)
- **Difference**: $x_1 - x_2$

### Key Python tools

| Task | Library / Function |
|---|---|
| One-hot encoding | `pandas.get_dummies()`, `sklearn.preprocessing.OneHotEncoder` |
| Ordinal encoding | `sklearn.preprocessing.OrdinalEncoder` |
| Target encoding | `sklearn.preprocessing.TargetEncoder` (scikit-learn >= 1.3), `category_encoders.TargetEncoder` |
| Binning | `pandas.cut()`, `pandas.qcut()`, `sklearn.preprocessing.KBinsDiscretizer` |
| Polynomial features | `sklearn.preprocessing.PolynomialFeatures` |
| Date extraction | `pandas.Series.dt` accessor (`.dt.month`, `.dt.dayofweek`, etc.) |

## Python implementation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures

# --- One-hot encoding ---
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' for linear models
encoded = ohe.fit_transform(df[['vehicle_type']])

# --- Ordinal encoding ---
oe = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['risk_tier_ord'] = oe.fit_transform(df[['risk_tier']])

# --- Target encoding with smoothing ---
te = TargetEncoder(smooth='auto')
df['zip_encoded'] = te.fit_transform(df[['zip_code']], df['claim_flag'])

# --- Binning ---
df['age_band'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
                         labels=['18-25', '26-35', '36-50', '51-65', '65+'])

# --- Date features ---
df['month'] = df['policy_start'].dt.month
df['day_of_week'] = df['policy_start'].dt.dayofweek
df['tenure_days'] = (pd.Timestamp.now() - df['policy_start']).dt.days

# --- Cyclical encoding ---
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# --- Interaction features ---
df['age_x_engine'] = df['driver_age'] * df['engine_size']

# --- Actuarial: exposure offset ---
df['exposure'] = df['days_in_force'] / 365.25
```

## R implementation (if applicable)

```r
library(recipes)

# recipes provides a clean pipeline for feature engineering
rec <- recipe(claim_flag ~ ., data = train_data) %>%
  step_dummy(vehicle_type, one_hot = FALSE) %>%       # one-hot (reference coding)
  step_ordinalscore(risk_tier) %>%                      # ordinal encoding
  step_discretize(age, num_breaks = 5) %>%              # binning
  step_interact(terms = ~ driver_age:engine_size) %>%   # interaction
  step_date(policy_start, features = c("month", "dow")) # date features

prepped <- prep(rec, training = train_data)
baked <- bake(prepped, new_data = test_data)
```

## Gotchas & lessons learned

1. **Target encoding leakage**: If you compute target-encoded values on the full training set and then use them in cross-validation, you have leaked information. Always compute target encoding within each CV fold, or use the built-in regularization (smoothing) parameter.

2. **One-hot encoding on test data**: If a category appears in the test set but not in the training set, your one-hot encoder will fail or silently produce wrong results. Use `handle_unknown='ignore'` in scikit-learn's `OneHotEncoder` to handle unseen categories gracefully.

3. **Dropping one category in one-hot encoding**: Linear models need you to drop one level to avoid perfect multicollinearity (the "dummy variable trap"). Tree-based models do not care -- they can handle the redundancy. Know which situation you are in.

4. **Binning destroys information**: Every time you bin a continuous variable, you lose granularity. A 25-year-old and a 35-year-old look identical if they are in the same bin. Use binning deliberately, not by default. In actuarial work, binning is traditional (rating tables), but modern ML models can often learn the non-linearities directly.

5. **Too many interaction features**: If you have 50 features and create all pairwise interactions, you now have 1,275 additional features. Most will be noise. Use domain knowledge to select meaningful interactions, or use tree-based models that learn interactions automatically.

6. **Cyclical encoding is often forgotten**: If you encode month as 1-12 without cyclical transforms, your model thinks December (12) is far from January (1). This can badly hurt seasonal models.

7. **Exposure must be an offset, not a regular feature**: In Poisson/frequency models, exposure should enter as `log(exposure)` offset, not as a regular predictor. This ensures that doubling the exposure doubles the expected count, which is the correct actuarial assumption.

8. **Feature engineering before or after train/test split**: Always split first, then engineer features. If you compute any statistic from the data (like target encoding means, or bin boundaries from quantiles), doing it on the full dataset before splitting leaks test information into training.

9. **Cardinality matters for encoding choice**: Rough rule of thumb:
   - 2-10 categories: one-hot encoding works fine
   - 10-50 categories: consider one-hot with possible grouping, or target encoding
   - 50+ categories: target encoding, hashing, or entity embeddings

10. **Do not forget the basics**: Simple features like ratios (`claims / exposure`), differences (`current_premium - prior_premium`), and aggregations (`average claim amount per policyholder`) are often the most powerful. Fancy transformations are no substitute for thinking about what drives your target variable.

## Resources

- **scikit-learn preprocessing documentation**: https://scikit-learn.org/stable/modules/preprocessing.html
- **category_encoders library**: https://contrib.scikit-learn.org/category_encoders/ -- implements dozens of encoding methods beyond what scikit-learn offers.
- **Feature Engineering and Selection (Kuhn & Johnson)**: Comprehensive textbook on the topic. Free online: https://bookdown.org/max/FES/
- **Kaggle Feature Engineering course**: https://www.kaggle.com/learn/feature-engineering -- practical, hands-on introduction.
- **Target encoding done right (Micci-Barreca, 2001)**: The original paper on target encoding with smoothing.
- **CAS Monograph on Predictive Modeling**: https://www.casact.org/ -- covers actuarial-specific feature engineering for insurance pricing.
