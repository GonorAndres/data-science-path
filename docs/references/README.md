# Academic References -- Insurance Pricing Project

## Downloaded PDFs

| File | Authors | Year | Title | Source |
|------|---------|------|-------|--------|
| `colella_jones_2023_ml_ratemaking_cas.pdf` | Colella & Jones | 2023 | Machine Learning and Ratemaking: Assessing Performance of Four Popular Algorithms for Modeling Auto Insurance Pure Premium | CAS E-Forum |
| `kuo_lupton_2023_explainability_insurance_pricing.pdf` | Kuo & Lupton | 2023 | Towards Explainability of Machine Learning Models in Insurance Pricing | Variance (via arXiv) |
| `mdpi_risks_2024_glm_vs_ml_motor_insurance.pdf` | MDPI Risks | 2024 | A Comparison of Generalised Linear Modelling with Machine Learning Approaches for Predicting Loss Cost in Motor Insurance | MDPI Risks 12(4) |

## Manual Download Required

| Authors | Year | Title | URL |
|---------|------|-------|-----|
| Noll, Salzmann & Wuthrich | 2020 | Case Study: French Motor Third-Party Liability Claims | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764 |

SSRN requires browser login to download. Save as `noll_salzmann_wuthrich_2020_french_motor_tpl.pdf` in this folder.

## Broader Curriculum References

The four PDFs above are pricing-specific. The broader theory and technique library draws on the books and papers below. PDFs are not redistributed in the repository (license); citations are kept here so any reader can locate them.

### Books

- Murphy, Kevin P. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022. Modern unified treatment of Bayesian methods and deep learning; the 2026 replacement for Bishop on most introductory shelves.
- Murphy, Kevin P. *Probabilistic Machine Learning: Advanced Topics*. MIT Press, 2023. Continuation covering reinforcement learning, deep generative models, and causal inference under one notation.
- Bishop, Christopher M. *Pattern Recognition and Machine Learning*. Springer, 2006. Classic; still the cleanest treatment of variational inference and graphical models for the Bayesian sections of the curriculum.
- Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. *Bayesian Data Analysis*. 3rd ed., CRC Press, 2013. Reference for every Bayesian essay and technique folder; hierarchical models chapter ties directly to credibility theory.
- Klugman, Stuart A., Harry H. Panjer, and Gordon E. Willmot. *Loss Models: From Data to Decisions*. 5th ed., Wiley, 2018. Canonical SOA STAM reference; the loss-distribution and credibility chapters anchor `loss-reserving` and `bayesian-credibility`.
- Wuthrich, Mario V., and Michael Merz. *Data Analytics for Non-Life Insurance Pricing*. Swiss Finance Institute, 2023. The actuarial / data-science bridge textbook; freely available online and the closest single-volume answer to "what does an actuary doing ML need to know?"
- Sutton, Richard S., and Andrew G. Barto. *Reinforcement Learning: An Introduction*. 2nd ed., MIT Press, 2018. Canonical RL reference; freely available online from the authors.
- Rasmussen, Carl Edward, and Christopher K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006. Free PDF at gaussianprocess.org; canonical GP reference and the foundation for the planned `techniques/gaussian-processes/` folder.
- McNeil, Alexander J., Rudiger Frey, and Paul Embrechts. *Quantitative Risk Management: Concepts, Techniques and Tools*. Revised ed., Princeton University Press, 2015. Extreme value theory, GARCH, copulas, and capital modeling for actuarial risk; the natural companion to `techniques/extreme-value-theory/` and the planned `techniques/garch-volatility/`.
- Angelopoulos, Anastasios N., and Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511, 2021. Free monograph; the practical entry point to conformal prediction for the planned uncertainty essay and technique folder.

### Papers (actuarial + DS bridge)

- Wuthrich, Mario V. "Machine Learning in Individual Claims Reserving." *Scandinavian Actuarial Journal*, 2018. Per-claim ML reserving; the modern alternative to chain-ladder triangle aggregation.
- Deprez, Philippe, Pavel V. Shevchenko, and Mario V. Wuthrich. "Machine Learning Techniques for Mortality Modeling." *European Actuarial Journal*, 2017. Tree ensembles and neural nets for mortality projection; bridges Lee-Carter to ML.
- Gabrielli, Andrea, and Mario V. Wuthrich. "An Individual Claims History Simulation Machine." *Risks*, 2018. Synthetic data generator for reserving research; useful for benchmarking ML reserving methods when proprietary data is unavailable.
- Mack, Thomas. "Distribution-Free Calculation of the Standard Error of Chain Ladder Reserve Estimates." *ASTIN Bulletin*, 23(2), 1993. Foundation of stochastic chain-ladder reserving; required reading for any `loss-reserving` extension.
