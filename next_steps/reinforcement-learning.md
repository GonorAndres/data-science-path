# Reinforcement Learning for Dynamic Optimization

## Purpose

Build agents that optimize sequential decisions over time. Unlike supervised ML (which predicts from a snapshot), reinforcement learning handles problems where actions today affect outcomes tomorrow. Insurance has natural RL applications that are almost entirely unexplored: pricing decisions that balance retention and profitability over a customer's lifetime, reserve adjustments that must be made sequentially as claim information develops, and settlement strategies where timing and negotiation have compounding effects. This is the most mathematically demanding specialization -- and the one where your actuarial training provides the strongest mathematical foundation.

---

## What It Unlocks

- **Career paths**: RL Engineer, Decision Intelligence Engineer, Optimization ML Engineer, Applied Research Scientist
- **Skills**: Markov decision processes, policy optimization, value estimation, exploration-exploitation tradeoffs, multi-armed bandits for A/B testing, simulation design, reward shaping
- **Domain leverage**: Insurance is full of sequential decision problems that the industry currently solves with static rules or heuristics. Applying RL here is genuine innovation, not incremental improvement.

---

## Key Technologies

| Technology | What It Does |
|---|---|
| **Stable-Baselines3** | Reliable, well-documented implementations of RL algorithms (PPO, A2C, SAC, DQN) in PyTorch. Best starting point. |
| **RLlib (Ray)** | Scalable RL library from Anyscale/Ray. Production-grade, supports multi-agent RL, distributed training. |
| **Gymnasium** | Standard API for RL environments (successor to OpenAI Gym). You will build custom insurance environments using this API. |
| **Multi-Armed Bandits** | Simplest RL formulation. Used for A/B testing, content personalization, and simple pricing optimization. |
| **Contextual Bandits** | Bandits with context (customer features). Maps directly to personalized pricing and offer selection. |
| **Policy Gradient Methods** | RL algorithms that directly optimize the policy. PPO (Proximal Policy Optimization) is the workhorse. |
| **Q-Learning / DQN** | Value-based RL methods. Learn the value of each state-action pair. Good for discrete action spaces (settle/litigate, approve/deny). |
| **Reward Shaping** | The art of designing reward signals that lead to desired agent behavior. Critical and often the hardest part of applied RL. |

---

## Project Ideas

### 1. Dynamic Pricing Agent

**Scope**: Build an RL agent that learns optimal insurance premiums by balancing retention (lower prices keep customers) against profitability (higher prices improve loss ratios). The agent observes customer features and market conditions, selects a price modifier, and learns from whether the customer renews and what their claims experience is.

**Dataset**: Build a custom Gymnasium environment simulating an insurance portfolio:
- Customer features: tenure, claim history, coverage type, price sensitivity (drawn from realistic distributions)
- Customer behavior model: probability of renewal as a function of price (price elasticity curves calibrated from literature)
- Claims model: frequency and severity drawn from actuarial distributions (Poisson-Gamma)
- Market dynamics: competitor pricing fluctuations, seasonal demand patterns
- Use the French Motor TPL dataset to calibrate realistic risk and pricing distributions

**Deliverables**:
- Custom Gymnasium environment implementing the insurance pricing simulation with configurable parameters
- Environment validation: verify that the simulation produces realistic loss ratios, retention rates, and portfolio growth patterns
- Baseline agents: random pricing, static actuarial pricing (GLM-based), simple rule-based (raise prices on bad risks, lower on good)
- RL agent: PPO-trained agent learning price modifiers. Action space: discrete price adjustments (-20% to +20% in 5% increments)
- Training analysis: reward curves, exploration behavior, convergence diagnostics
- Policy analysis: what pricing strategy did the agent learn? Compare its decisions against actuarial intuition by customer segment
- Sensitivity analysis: how does the optimal policy change under different assumptions about price elasticity and claim severity?
- Reward shaping exploration: compare agents trained with different reward formulations (profit only, profit + retention bonus, profit with risk-adjusted capital charge)
- Write-up discussing the gap between RL-optimal pricing and regulatory constraints (filed rates, non-discrimination requirements)

**Complexity**: High. Building a realistic simulation environment is the hardest part. The RL training itself is straightforward once the environment is right.

### 2. Optimal Reserve Setting Agent

**Scope**: Build an RL agent that makes sequential case reserve adjustments as information about a claim develops over time. At each time step, the agent observes new information (medical reports, legal updates, adjuster notes) and must decide how to adjust the reserve. The goal is to minimize reserve volatility while maintaining accuracy.

**Dataset**: Build a custom environment simulating claim development:
- Claims with known ultimate values (generated from actuarial development patterns)
- Information revealed over time: initial report, first medical assessment, legal representation notification, independent medical exam, settlement negotiation
- Reserve accuracy metric: deviation between current reserve and ultimate outcome
- Use triangle development factors from published CAS papers to calibrate realistic development patterns
- Different claim types (bodily injury, property damage, workers' comp) with distinct development characteristics

**Deliverables**:
- Custom Gymnasium environment simulating claim lifecycle with sequential information revelation
- State representation: current reserve, elapsed time, claim features, information received so far, portfolio context (total reserves, calendar year pressure)
- Action space: reserve adjustment decisions (increase/decrease by percentage or absolute amount, or hold)
- Baseline agents: chain-ladder mechanical reserves, case adjuster heuristics (initial + fixed development factors)
- RL agent: DQN or PPO trained to minimize a compound objective (reserve accuracy + stability + calendar year impact)
- Comparison against actuarial methods: does the RL agent develop reserves more accurately than traditional chain-ladder or BF methods?
- Analysis of learned behavior: does the agent exhibit recognizable actuarial patterns (e.g., conservative initial reserves, gradual release)?
- Stress testing: how does the agent perform on tail claims (large, complex, long-developing)?

**Complexity**: Very high. This requires deep actuarial knowledge to build a credible simulation. The reward function design is particularly challenging -- reserving has multiple competing objectives.

### 3. Claim Settlement Strategy Agent

**Scope**: Build an RL agent that learns when to make settlement offers, how much to offer, and when to escalate to litigation. The agent balances settlement cost against litigation risk, legal fees, and time value of money.

**Dataset**: Build a custom environment simulating claim negotiation:
- Claim features: liability assessment, injury severity, jurisdiction, claimant representation, policy limits
- Claimant behavior model: probability of accepting offer as a function of offer amount and time elapsed (based on negotiation theory)
- Litigation model: probability of various trial outcomes, legal cost accumulation over time, trial duration distribution
- Time value of money and expense loading
- Calibrate using published jury verdict databases and defense cost studies

**Deliverables**:
- Custom Gymnasium environment implementing the settlement negotiation simulation
- State: claim features, time elapsed, offers made, claimant responses, accumulated legal costs, policy remaining limits
- Action space: make offer (at various amounts), request mediation, proceed to trial, delay
- Baseline agents: fixed-formula settlement (e.g., 3x medical specials), adjuster heuristics, always-litigate, always-settle
- RL agent trained with PPO on the compound objective: minimize total claim cost (settlement/verdict + legal fees + time cost)
- Strategy analysis: what patterns does the agent learn? Does it make early generous offers for strong-liability claims and fight weak claims?
- Sensitivity to claimant behavior model: how does optimal strategy change if claimants are more/less litigious?
- Ethical analysis: discuss limitations and guardrails needed (agent should not learn to exploit vulnerable claimants)
- Write-up framing this as a decision-support tool (augmenting adjusters, not replacing them)

**Complexity**: Very high. The claimant behavior model is the critical assumption. Ethical considerations require careful treatment.

---

## Career Impact & Salary Context

- **Demand**: RL is a rare skill in industry. Most companies that use it are at the frontier (DeepMind, OpenAI, Waymo, quantitative trading firms, robotics companies). Insurance applications are almost entirely greenfield.
- **Where to target**: Quantitative trading and market-making firms (Citadel, Two Sigma, Jane Street -- they use RL for execution optimization), pricing optimization teams at tech companies (Uber, Airbnb, Amazon), robotics companies, research labs, and forward-thinking insurance companies building next-generation decision systems.
- **Salary range (US)**: $160K-$250K for RL engineers. Research scientist roles (where RL is most common) can reach $300K+ at top firms. The rarity premium is significant.
- **Mexico/LATAM remote**: $55K-$100K USD for remote roles. Limited RL roles in LATAM, but global remote positions are accessible with a strong portfolio.
- **Portfolio differentiation**: An RL project applied to insurance is extremely unusual. Interview panels will remember it because nobody else is doing this.

---

## How Your Actuarial Background Provides an Edge

- **Mathematical foundation**: RL is built on probability theory, dynamic programming, and stochastic processes -- all core actuarial curriculum. The Bellman equation is a stochastic recursion, which you have seen in ruin theory and credibility.
- **Simulation expertise**: Actuarial work involves heavy simulation (Monte Carlo for capital modeling, stochastic reserving). Building RL environments requires exactly this skill.
- **Domain knowledge for reward design**: The hardest part of applied RL is designing the reward function. Your actuarial training lets you define actuarially meaningful objectives (combined ratio optimization, reserve adequacy within tolerance, surplus maximization).
- **Understanding tradeoffs**: Actuaries routinely balance competing objectives (price competitiveness vs. adequacy, reserve conservatism vs. calendar year results). RL formalizes these as multi-objective optimization problems.
- **Credibility theory connection**: Credibility theory is about optimal weighting of individual experience vs. class experience as data accumulates. This is conceptually related to exploration-exploitation tradeoffs in bandits.
- **Risk-adjusted thinking**: Actuaries think in terms of risk-adjusted returns, not just expected returns. This translates directly to risk-sensitive RL formulations.

---

## Prerequisites

- **Phase 1** (Python ML Foundations): Must be complete -- you need strong Python and neural network basics.
- **Phase 5** (Statistics & Experimentation): Recommended -- bandit algorithms are the bridge between A/B testing and full RL.
- Strong mathematical foundations in probability and stochastic processes (you have this from actuarial training).
- Basic PyTorch proficiency (neural network training loops, gradient computation).
- Comfort with simulation and Monte Carlo methods.

---

## Estimated Timeline

- **Weeks 1-2**: RL fundamentals (MDPs, Bellman equations, dynamic programming). Work through Sutton & Barto chapters 1-6.
- **Weeks 3-4**: Multi-armed bandits and contextual bandits. Implement from scratch, then use library implementations.
- **Weeks 5-6**: Deep RL (DQN, PPO, A2C). Train agents on standard Gymnasium environments (CartPole, LunarLander).
- **Weeks 7-9**: Project 1 (Dynamic Pricing Agent) -- environment design is the focus.
- **Weeks 10-12**: Project 2 (Optimal Reserve Setting) -- deep actuarial domain modeling.
- **Weeks 13-15**: Project 3 (Claim Settlement Strategy) -- multi-objective optimization and ethical considerations.
- **Week 16**: Integration, portfolio write-up, connecting RL to actuarial decision theory.
- **Total: 14-16 weeks** at 15-20 hours/week

---

## Resources

### Courses
- [Sutton & Barto: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) -- the textbook, free online. Non-negotiable reading.
- [David Silver's RL Course (UCL/DeepMind)](https://www.davidsilver.uk/teaching/) -- excellent video lectures
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) -- practical, hands-on, free
- [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/) -- rigorous treatment

### Papers
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017) -- PPO, the workhorse algorithm
- "Human-level Control through Deep Reinforcement Learning" (Mnih et al., 2015) -- the DQN paper
- "A Contextual-Bandit Approach to Personalized News Article Recommendation" (Li et al., 2010) -- contextual bandits applied
- "Dynamic Pricing with Reinforcement Learning" (various) -- survey the literature on RL pricing

### Repos & Tools
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) -- reliable RL implementations
- [RLlib (Ray)](https://github.com/ray-project/ray/tree/master/rllib) -- scalable RL
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) -- environment API
- [CleanRL](https://github.com/vwxyzjn/cleanrl) -- single-file RL implementations, excellent for learning
- [Tianshou](https://github.com/thu-ml/tianshou) -- modular RL library with good documentation
