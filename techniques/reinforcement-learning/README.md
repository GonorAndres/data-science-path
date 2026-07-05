---
tags: [reinforcement-learning, sequential-decision, deep-learning]
status: draft
created: 2026-04-25
updated: 2026-04-25
---

# Reinforcement Learning

## What it is

Reinforcement learning (RL) is the framework for sequential decision-making under uncertainty. An agent observes a state, takes an action, receives a reward, transitions to a new state, and repeats. The goal is to learn a policy: a mapping from states to actions that maximizes expected cumulative reward over time.

RL generalizes the multi-armed bandit (one-shot decisions) and connects to optimal control, dynamic programming, and decision theory. In 2026 it powers recommendation systems, dialogue managers, robotics, autonomous driving research, AlphaFold-style protein design, and the alignment phase of frontier LLMs (RLHF, RLAIF, DPO).

## The Markov decision process

A finite MDP is the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: the state space.
- $\mathcal{A}$: the action space.
- $P(s' \mid s, a)$: the transition kernel, the probability of moving to $s'$ from $s$ under action $a$.
- $R(s, a)$: the immediate reward (or its expectation) when taking $a$ in $s$.
- $\gamma \in [0, 1)$: the discount factor, controlling how much future rewards count relative to immediate ones.

The **policy** $\pi(a \mid s)$ is a (stochastic) mapping from states to actions. The **value function** of a policy is the expected discounted return starting from $s$:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[\, \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \,\Big|\, s_0 = s \,\right]
$$

The **action-value (Q) function** is the expected return from taking action $a$ first and following $\pi$ thereafter:

$$
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\, \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \,\Big|\, s_0 = s, a_0 = a \,\right]
$$

The Markov property ($s_{t+1}$ depends on the past only through $s_t$ and $a_t$) is what makes the framework tractable; if it does not hold, augment the state with enough history to make it hold.

## Bellman equations

The value function satisfies a recursive equation, the **Bellman equation**:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \right]
$$

The optimal value function $V^*$ satisfies the **Bellman optimality equation**:

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]
$$

Two classical algorithms for solving small MDPs when $P$ and $R$ are known:

- **Value iteration.** Initialize $V$ arbitrarily; repeatedly apply the Bellman optimality update until convergence.
- **Policy iteration.** Alternate between policy evaluation (solve the linear system for $V^\pi$) and policy improvement (greedy with respect to $V^\pi$).

Both converge to the optimum in a finite number of iterations on finite MDPs.

## Q-learning

When $P$ and $R$ are unknown, the agent must learn from interaction. **Q-learning** (Watkins 1989) maintains an estimate $Q(s, a)$ and updates it from observed transitions $(s, a, r, s')$:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

The update target $r + \gamma \max_{a'} Q(s', a')$ is the temporal-difference (TD) target. Q-learning is **off-policy**: the policy used to gather data (often $\epsilon$-greedy) does not have to match the policy being learned (greedy in $Q$).

Convergence conditions:

- Every state-action pair must be visited infinitely often.
- The learning rate must satisfy $\sum \alpha_t = \infty$ and $\sum \alpha_t^2 < \infty$.
- Rewards bounded.

For tabular Q-learning these conditions guarantee convergence to $Q^*$ with probability 1.

## Deep Q-Networks (DQN)

When the state space is too large for a tabular Q-function (raw pixels, robot proprioception), approximate $Q(s, a; \theta)$ with a neural network. Naive substitution is unstable; **DQN** (Mnih et al. 2015) introduced two innovations that fixed the instability and enabled human-level Atari play:

1. **Experience replay.** Store transitions in a buffer $\mathcal{D}$; train on uniform random mini-batches from $\mathcal{D}$ rather than on the most recent transition. This breaks correlation between consecutive samples and reuses data many times, both of which stabilize training.
2. **Target network.** Maintain a separate, slowly updated copy of the Q-network, $Q_{\theta^-}$, used to compute the TD target. Copy the online network weights to the target every $N$ steps. This breaks the feedback loop where bootstrapping a moving target diverges.

Loss:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$

Subsequent improvements: Double DQN (decouples action selection from evaluation), Dueling DQN (separates state value from action advantage), Prioritized Experience Replay (sample transitions with high TD error more often), Rainbow (combines all of the above).

## Policy gradient methods

Q-learning learns a value function and derives the policy. Policy gradient methods learn the policy $\pi_\theta(a \mid s)$ directly by gradient ascent on the expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_t \gamma^t R(s_t, a_t) \right]
$$

The **policy gradient theorem** gives the gradient in a form computable from samples:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

where $G_t = \sum_{k \geq t} \gamma^{k - t} R(s_k, a_k)$ is the return-to-go.

**REINFORCE** (Williams 1992) is the literal implementation: roll out an episode, compute returns, take a gradient step. It works but has very high variance because $G_t$ varies wildly across rollouts.

**Variance reduction via a baseline.** Subtract a state-dependent baseline $b(s_t)$, typically a learned value function $V_\phi(s_t)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - V_\phi(s_t)) \right]
$$

Subtracting $b(s_t)$ does not change the gradient's expectation but dramatically reduces its variance. This is the entry point to actor-critic methods.

## Actor-critic and PPO

In actor-critic the policy ($\pi_\theta$, the actor) and a value function ($V_\phi$, the critic) are learned jointly. The advantage $A_t = G_t - V_\phi(s_t)$ replaces $G_t$ in the policy gradient.

- **A2C / A3C** (Mnih et al. 2016). Synchronous and asynchronous actor-critic. Strong baselines but sensitive to hyperparameters.
- **PPO** (Schulman et al. 2017). The dominant practical RL algorithm in 2026. PPO modifies the policy gradient with a **clipped objective** that prevents the new policy from moving too far from the old one in a single update:

$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\!\big( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \big) \right]
$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ and $\epsilon \approx 0.2$. The clip prevents catastrophic policy collapse, the most common failure mode in vanilla policy gradient.

PPO additionally adds an entropy bonus to encourage exploration:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{PPO}} - c_1 \mathcal{L}^{\text{value}} + c_2 \mathcal{H}[\pi_\theta]
$$

The combination of clipped objective and entropy bonus makes PPO the default for continuous control, game-playing, and the RLHF stage of LLM training.

## Applications in data science

### Recommendation as RL

Frame the recommender as an agent: state is the user's interaction history, action is the next item to show, reward is engagement (click, dwell, conversion). RL captures the sequential nature better than supervised next-click prediction because it accounts for **long-term engagement** (recommending a clickbait item now hurts future sessions). Slate-MDP frameworks (Ie et al. 2019) extend this to ranking the slate, not just a single item.

### Hyperparameter tuning as RL (AutoRL)

Pose the hyperparameter search as an MDP where the state is the current configuration plus partial training metrics, the action is "change hyperparameter $h$ to value $v$", and the reward is the validation metric. AutoML systems use RL controllers (or Bayesian optimization, see `techniques/hyperparameter-optimization-optuna`) to learn search policies that beat random search.

### LLM fine-tuning via RLHF and DPO

The third stage of frontier LLM training. Given a supervised fine-tuned model and a reward model trained on human preferences, run PPO with the LLM as the actor and the reward model as the environment to align the model with human-preferred outputs. PPO is finicky here; **DPO** (Rafailov et al. 2023) reformulates the same objective without an explicit reward model and has become the practical default for instruction-tuning small to mid-size LLMs.

### Operations research applications

Inventory management with stochastic demand, dynamic pricing under competition, and reinsurance program optimization can all be cast as MDPs. The transition from research to deployment is gated by the difficulty of getting reliable reward signals from a simulator that matches reality.

## Common gotchas

1. **Reward hacking.** Agents exploit the literal reward function in unintended ways. The classic example is a robot learning to flip a pancake by maximizing time-in-air: it just throws the pancake. Reward shaping must be done with care; auditing the policy's actual behavior beats reading reward curves.
2. **Off-policy distribution shift.** Replay-buffer methods drift when the policy changes faster than the buffer turns over. Symptoms: the loss decreases but evaluation performance regresses.
3. **Sparse rewards.** Most realistic problems give zero reward 99 percent of the time, then a final +1 or -1. Vanilla policy gradient cannot learn under this. Use reward shaping, hierarchical RL, or curriculum learning.
4. **Sim-to-real gap.** Policies trained in a simulator transfer poorly to physical systems because the simulator is wrong. Domain randomization (train across many simulator variants) is the standard mitigation.
5. **Sample efficiency.** Model-free RL is famously sample-inefficient. For real-world problems with expensive samples (clinical trials, real customers) use offline RL, model-based RL, or contextual bandits if the sequential structure is shallow.

## Resources

- `techniques/multi-armed-bandits/README.md` for the simpler one-shot setting.
- `next_steps/reinforcement-learning.md` for project ideas tailored to this portfolio.
- `docs/theory/09_llm_foundations_2026.md` for the RLHF context.

## References

- Sutton, Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press 2018 (free online at incompleteideas.net).
- Mnih et al., "Human-level control through deep reinforcement learning" (DQN), Nature 2015.
- Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms" (PPO), 2017.
- Rafailov et al., "Direct Preference Optimization" (DPO), NeurIPS 2023.
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (REINFORCE), Machine Learning 1992.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), NeurIPS 2022.
- OpenAI Spinning Up (spinningup.openai.com): the practitioner's introduction to deep RL with code.
