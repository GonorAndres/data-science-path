# Churn Modeling

## What it is

Churn modeling is the practice of predicting which customers are likely to stop doing business with you -- and ideally, predicting it early enough that you can do something about it. "Churn" can mean canceling a subscription, not renewing a policy, closing an account, or simply going silent. The model typically outputs a churn probability for each customer over a defined time window, enabling the business to target retention efforts at those most likely to leave (and most valuable to keep). It is one of the highest-ROI applications of data science because retaining an existing customer is almost always cheaper than acquiring a new one.

## Why it matters

- **The economics are clear**: Acquiring a new customer costs 5-25x more than retaining an existing one (the exact ratio varies by industry, but the direction is universal). Even a small reduction in churn -- say 2 percentage points -- can translate into millions in preserved revenue.
- **Insurance relevance**: In property and casualty insurance, policy lapse/non-renewal is the primary form of churn. Retention rates directly drive combined ratios, loss experience stability, and long-term profitability. A policyholder who leaves after one year has not yet generated enough premium to cover acquisition costs, commissions, and the fixed costs of underwriting.
- **Proactive vs. reactive**: Without a churn model, you find out a customer left after it happens. With a model, you can intervene weeks or months before the decision is made -- with a targeted offer, a service recovery, or a proactive call from an agent.
- **Feeds into CLV**: Churn probability is one of the key inputs to customer lifetime value (CLV) models. Better churn prediction leads to better CLV estimates, which leads to better resource allocation.
- **Feature engineering laboratory**: Churn modeling is an excellent way to learn feature engineering. The features that predict churn (behavioral changes, engagement drops, complaint patterns) teach you to think about leading indicators rather than lagging ones.

## When to use it

- You have a contractual business (subscriptions, insurance policies) and want to predict non-renewal or cancellation.
- You have a non-contractual business and want to flag customers who are likely to become inactive.
- You are building a retention program and need to prioritize which customers to contact.
- You want to estimate the financial impact of retention initiatives (combine churn probability with CLV).
- You are developing an early-warning system for customer health scoring.
- You need to understand the key drivers of churn to inform product or service improvements.
- You are designing a pricing strategy and want to understand the retention elasticity of price changes.

## How it works (intuitive)

### The health checkup analogy

Think of churn modeling like a doctor's preventive health screening. The doctor does not wait for a heart attack -- they measure blood pressure, cholesterol, weight, family history, and lifestyle factors to estimate your risk. Patients with high-risk scores get interventions (medication, diet changes) before the event happens.

Churn modeling does the same for customers:
1. **Collect symptoms** (features): declining engagement, fewer logins, support tickets, payment delays.
2. **Build a risk model**: Train a classifier that maps symptoms to churn probability.
3. **Screen customers**: Score the entire customer base regularly.
4. **Intervene on high-risk customers**: Target retention offers, proactive outreach, or service recovery.

### Step-by-step process

**Step 1: Define churn**

This is the most important and most underestimated step. Churn is not always obvious:

| Business type | Churn definition | Nuances |
|---------------|-----------------|---------|
| Subscription SaaS | Canceled their subscription | Clear event, easy to observe |
| Insurance | Did not renew the policy | Need to define the "grace period" for late renewals |
| E-commerce | No purchase in X days | Must choose X; too short = false positives, too long = too late to act |
| Banking | Closed all accounts | Some customers go dormant but do not close |
| Mobile app | No login in X days | Similar to e-commerce; requires defining the inactivity threshold |

For non-contractual settings, you must choose an **inactivity threshold**. A common approach: define churn as "no transaction in the past 90 days" (or whatever period matches your business cycle). This choice directly affects your target variable and model performance.

**Step 2: Define the prediction window**

You need to be specific about the time horizon:
- **Observation period**: The historical window from which you compute features (e.g., "the past 12 months of behavior").
- **Gap period**: A buffer between the observation window and the prediction window, to ensure you are not using information that would not be available at prediction time (e.g., 1-month gap).
- **Prediction window**: The future window during which churn may occur (e.g., "will this customer churn in the next 3 months?").

```
|------ Observation Period ------|-- Gap --|------ Prediction Window ------|
         Features computed here                   Churn label defined here
```

The gap period is critical. Without it, you risk data leakage: using information about early churn signals that you would not have when deploying the model.

**Step 3: Engineer features**

Churn features generally fall into these categories:

| Category | Examples |
|----------|---------|
| **Tenure** | Months since first purchase, policy year |
| **Behavioral (recent)** | Transactions in last 30/60/90 days, login frequency, feature usage |
| **Behavioral (trends)** | Change in purchase frequency month-over-month, declining engagement score |
| **Financial** | Total spend, average order value, premium amount, payment method |
| **Service interactions** | Number of support tickets, complaint flag, NPS score, claim frequency |
| **Contractual** | Contract end date proximity, auto-renewal status, discount/promotion history |
| **Demographic** | Age, tenure, geography, segment |
| **Competitive signals** | Price comparison site visits (if tracked), quote requests from competitors |

The most predictive features are usually **behavioral trends** -- not the absolute level of engagement, but the *change* in engagement. A customer who went from 10 logins per week to 2 is more at risk than one who has always logged in twice per week.

**Step 4: Handle class imbalance**

Churn is typically a rare event (5-15% in many industries). This creates class imbalance, which can cause naive models to predict "no churn" for everyone and still achieve high accuracy.

Strategies:
- **Resampling**: Oversample the minority class (SMOTE, random oversampling) or undersample the majority class.
- **Class weights**: Set `class_weight='balanced'` in scikit-learn classifiers to penalize misclassification of churners more heavily.
- **Threshold tuning**: Instead of using the default 0.5 probability threshold, choose a threshold that optimizes for the business metric (e.g., maximize the F1 score or a custom cost function).
- **Evaluation metrics**: Do not use accuracy. Use precision, recall, F1, AUC-ROC, or AUC-PR (precision-recall curve, which is more informative for imbalanced classes).

**Step 5: Train the model**

Common model choices for churn:

| Model | Pros | Cons |
|-------|------|------|
| Logistic regression | Interpretable, fast, baseline | Linear decision boundary |
| Random forest | Handles non-linearities, feature importance | Less interpretable |
| Gradient boosting (XGBoost, LightGBM) | Often highest AUC, handles mixed feature types | More complex, requires tuning |
| Survival models (Cox PH, AFT) | Models time-to-churn, handles censoring naturally | Different output (hazard, not probability at a fixed horizon) |
| Neural networks | Can capture complex patterns | Overkill for most churn problems, less interpretable |

Gradient boosting (XGBoost or LightGBM) is the pragmatic default for most churn modeling projects.

**Step 6: Evaluate and calibrate**

- **AUC-ROC**: How well the model ranks churners above non-churners. Above 0.75 is usually acceptable; above 0.85 is strong.
- **AUC-PR**: More informative than AUC-ROC when the positive class is rare.
- **Calibration**: The predicted probabilities should match observed churn rates. A customer predicted at 30% churn probability should actually churn about 30% of the time. Use calibration plots and Platt scaling or isotonic regression if needed.
- **Lift chart**: How much better is the model than random targeting? If the top decile (10% of customers by predicted churn probability) contains 40% of actual churners, the model has 4x lift.
- **Economic evaluation**: Ultimately, the model is only useful if acting on its predictions is profitable. Define the cost of intervention, the success rate of retention offers, and the value of retained customers. Then compute the expected profit from using the model.

**Step 7: Deploy and monitor**

- Score customers regularly (weekly or monthly).
- Feed scores into CRM or agent dashboards.
- Track actual churn rates against predicted rates to detect model drift.
- Retrain the model periodically (quarterly or when performance degrades).

### The smoke detector analogy

A churn model is like a smoke detector. It does not prevent fires -- it alerts you early enough to act. The key trade-off is:
- **Too sensitive** (low threshold): many false alarms (non-churners flagged as at-risk), which wastes retention budget.
- **Not sensitive enough** (high threshold): misses actual churners, defeating the purpose.

The right threshold depends on the relative cost of false positives (wasted retention offer) vs. false negatives (lost customer).

## Key concepts

### Binary classification formulation

$$y_i = \begin{cases} 1 & \text{if customer } i \text{ churns within the prediction window} \\ 0 & \text{otherwise} \end{cases}$$

$$P(\text{churn}_i | \mathbf{x}_i) = f(\mathbf{x}_i; \theta)$$

where $\mathbf{x}_i$ is the feature vector and $\theta$ are learned parameters.

### Logistic regression for churn

$$P(\text{churn} | \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p)}}$$

The coefficients $\beta_j$ have an interpretation: $e^{\beta_j}$ is the odds ratio for a one-unit increase in feature $x_j$. This interpretability makes logistic regression valuable for understanding churn drivers even when you use a more complex model for prediction.

### Survival-based churn modeling

Instead of predicting binary churn within a fixed window, survival models predict the *time until churn*:

$$h(t | \mathbf{x}) = h_0(t) \cdot e^{\beta_1 x_1 + \cdots + \beta_p x_p}$$

This is the Cox proportional hazards model. The hazard function $h(t)$ represents the instantaneous risk of churn at time $t$, given survival to time $t$.

**Advantages of survival models**:
- Handle censoring: customers who have not yet churned are right-censored, not discarded.
- Predict churn at any time horizon, not just a fixed one.
- Model the entire churn trajectory, not just a binary outcome.
- Natural connection to actuarial decrement models.

**Disadvantage**: Slightly more complex to implement and explain to stakeholders.

### Class imbalance metrics

**Precision**: Of all customers the model flagged as "will churn," what fraction actually churned?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)**: Of all customers who actually churned, what fraction did the model catch?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score**: Harmonic mean of precision and recall.

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**AUC-PR**: Area under the precision-recall curve. More informative than AUC-ROC for rare events.

### Cost-sensitive evaluation

$$\text{Expected Profit per Customer} = P(\text{churn}) \cdot [P(\text{save} | \text{intervene}) \cdot CLV - C_{\text{intervention}}] - (1 - P(\text{churn})) \cdot C_{\text{unnecessary intervention}}$$

This framework tells you the optimal probability threshold: intervene when the expected profit of intervention exceeds the expected cost.

### Feature engineering patterns

**Ratio features**: $\frac{\text{activity last 30 days}}{\text{activity last 90 days}}$ captures declining engagement.

**Trend features**: Slope of a linear regression fit to weekly activity over the past 3 months.

**Time-since features**: Days since last login, days since last purchase, days since last support ticket.

**Aggregation features**: Mean, std, min, max of transaction amounts over different windows.

**Interaction features**: Tenure $\times$ recent activity (long-tenured but recently disengaged = high risk).

## Python implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# --- Step 1: Prepare features and target ---
# Assume df has customer-level features and a binary 'churned' column
X = df.drop(columns=['customer_id', 'churned'])
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 2: Handle class imbalance (optional) ---
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# --- Step 3: Train XGBoost ---
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    eval_metric='aucpr',
    random_state=42
)
model.fit(X_train, y_train)  # or X_train_res, y_train_res if using SMOTE

# --- Step 4: Evaluate ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"AUC-PR: {average_precision_score(y_test, y_pred_proba):.4f}")

# Calibration
calibrated_model = CalibratedClassifierCV(model, cv=5, method='isotonic')
calibrated_model.fit(X_train, y_train)

# --- Step 5: Feature importance ---
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# --- Survival-based churn with lifelines ---
from lifelines import CoxPHFitter

cox = CoxPHFitter(penalizer=0.01)
cox.fit(
    survival_df,  # columns: duration, event (1=churned, 0=censored), + features
    duration_col='tenure_months',
    event_col='churned'
)
cox.print_summary()
cox.plot()
```

Key libraries:
- **scikit-learn**: Logistic regression, random forest, evaluation metrics, calibration.
- **xgboost** / **lightgbm**: Gradient boosting classifiers.
- **imbalanced-learn**: SMOTE and other resampling techniques.
- **lifelines**: Survival models (Cox PH, Kaplan-Meier) for time-to-churn.
- **shap**: Model interpretation (see `techniques/shap-interpretability/`).

## R implementation (if applicable)

```r
library(caret)
library(xgboost)
library(survival)
library(pROC)

# Binary classification with caret
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                     summaryFunction = twoClassSummary, sampling = "smote")

model <- train(churned ~ ., data = train_df, method = "xgbTree",
               trControl = ctrl, metric = "ROC")

# Survival model
surv_model <- coxph(Surv(tenure_months, churned) ~ age + premium + claims_count,
                    data = surv_df)
summary(surv_model)
```

**Python vs. R tradeoff**: Python's scikit-learn + XGBoost pipeline is more production-oriented with easier deployment. R's `survival` package is the gold standard for survival models and has more diagnostic tools. For the classification approach, Python is usually preferred; for survival-based churn, R is competitive.

## Gotchas & lessons learned

### 1. Churn definition is your most important decision
An ambiguous or wrong churn definition will poison everything downstream. In insurance, is a "churn" customer one who did not renew, or one who did not renew and did not come back within 60 days? For non-contractual businesses, the inactivity threshold (30 days? 90 days? 180 days?) determines the positive class. Test multiple definitions and see how they affect model performance and business actions.

### 2. Data leakage is everywhere
The most common form: including features that are only known *because* the customer churned. For example, "cancellation reason" is not a predictor -- it is a consequence. The observation-gap-prediction window framework prevents most leakage, but audit your features carefully.

### 3. Do not optimize for accuracy
With 5% churn, a model that predicts "no churn" for everyone is 95% accurate. Use AUC-ROC, AUC-PR, F1, or (best of all) a custom cost function that reflects the business economics.

### 4. Threshold choice is a business decision
The default 0.5 threshold is almost never optimal for churn. If retention offers are cheap and customer value is high, use a lower threshold (catch more churners, accept more false positives). If offers are expensive, use a higher threshold.

### 5. SMOTE can hurt more than it helps
SMOTE creates synthetic minority examples by interpolating between existing ones. This can introduce noise, especially with high-dimensional or categorical features. Try multiple approaches: SMOTE, class weights, threshold tuning, no adjustment. Let the evaluation metrics decide.

### 6. Behavioral features dominate
In almost every churn model, behavioral features (engagement changes, usage patterns) outperform demographic features (age, gender, geography). Invest your feature engineering time in behavioral signals.

### 7. The "save rate" matters
Even a perfect churn model is useless if your retention interventions do not work. If you predict 100 churners and contact all of them, but your offer only saves 10%, you have saved 10 customers. Factor the save rate into your ROI calculations.

### 8. Beware of voluntary vs. involuntary churn
Voluntary churn (customer chooses to leave) and involuntary churn (payment failure, account suspended) have different causes and different solutions. Mixing them into one model can reduce predictive power. Consider building separate models or at least separate features for each type.

### 9. Model decay is real
Customer behavior changes over time (new competitors, economic shifts, product changes). A churn model trained on 2023 data may degrade by mid-2024. Monitor performance continuously and retrain on a schedule.

### 10. The survival approach is underused
Most teams default to binary classification because it is familiar. But survival models are often more natural for churn because they handle censoring, model time dynamics, and can predict churn at any horizon. If your churn has a strong time component (e.g., most churn happens at contract end), consider the survival approach.

## Resources

- **Scikit-learn imbalanced classification tutorial**: [sklearn docs](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
- **imbalanced-learn**: [imbalanced-learn docs](https://imbalanced-learn.org/) -- SMOTE and other resampling methods.
- **XGBoost documentation**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- **lifelines**: [lifelines docs](https://lifetimes.readthedocs.io/) -- for survival-based churn.
- **SHAP**: [shap docs](https://shap.readthedocs.io/) -- for interpreting churn model features.
- **Kaggle**: The Telco Customer Churn dataset is the standard practice dataset.
- **Provost & Fawcett**: "Data Science for Business" -- Chapter 8 covers churn modeling as a case study with cost-sensitive evaluation.
- **DataCamp**: "Machine Learning for Marketing in Python" and "Customer Analytics and A/B Testing in Python."
- **Actuarial application**: For the survival analysis connection, see the `techniques/survival-analysis/` folder in this repo.
