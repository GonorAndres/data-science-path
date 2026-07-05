# Problem Framing: Uplift Modeling for Marketing

## The paradox of targeting

A naive targeting model predicts "probability of purchase" and targets the top N. This is wrong. Some high-probability buyers would buy anyway; the campaign wastes money on them. Some low-probability buyers cannot be persuaded; the campaign wastes money on them too.

What we actually want: customers whose probability of purchase increases with the treatment. "Persuadables."

Formally:

$$
\tau(x) = E[Y | X = x, T = 1] - E[Y | X = x, T = 0]
$$

is the **Conditional Average Treatment Effect (CATE)**. Target the top N by $\hat{\tau}(x)$.

## Four segments

For each individual, the treatment has one of four effects:

- **Persuadables.** Buy only if treated. Target them.
- **Sure things.** Buy regardless. Do not waste treatment on them.
- **Lost causes.** Do not buy regardless. Do not waste treatment on them.
- **Sleeping dogs.** Buy only if NOT treated. Avoid treating them (campaign backfires).

Classical response models confuse these segments. Uplift modeling separates them.

## Stakeholders

- **Marketing.** Wants to spend campaign budget efficiently.
- **Finance.** Wants ROI tracked, not vanity "click-through rate".
- **Product.** Sometimes wants behavioral insight beyond targeting.

## Non-goals

- Not computing ATE from a non-randomized experiment (requires different methods; see `docs/theory/08_experimentation_causal_primer.md`).
- Not optimizing spend across multiple campaigns (a separate portfolio-optimization problem).
- Not personalizing message content (just treatment/no-treatment).

## Success metrics

- **Qini coefficient.** Area under the Qini curve. Higher is better.
- **Uplift at top 10 percent.** Expected incremental purchases per 100 targets.
- **Cost per incremental conversion.** Business metric integrating uplift with campaign cost.

## Data requirements

Uplift modeling needs:

- Features observable before treatment.
- Treatment assignment (ideally randomized; observational requires strong assumptions).
- Outcome observable after treatment.

Criteo Uplift dataset has all three, with anonymized features, making it ideal for study.

## Cross-links

- `docs/theory/08_experimentation_causal_primer.md`
- `01_potential_outcomes.md`
