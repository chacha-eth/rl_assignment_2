import numpy as np
import matplotlib.pyplot as plt
from epilson_greedy import EpsilonGreedy
from ucb import UCB
from thomson_sampling import ThompsonSampling


# Experiment Settings
num_arms = 5
true_rewards = np.random.rand(num_arms) * 10  # Mean reward between 0 and 10
true_probs = np.random.rand(num_arms)  # Bernoulli probabilities
steps = 1000

epsilon_values = [0.1, 0.01, 0.5]
c_values = [1, 2, 5]

# Run experiments
eps_results = [EpsilonGreedy(num_arms, true_rewards, eps).run(steps) for eps in epsilon_values]
ucb_results = [UCB(num_arms, true_rewards, c).run(steps) for c in c_values]
thompson_result = ThompsonSampling(num_arms, true_probs).run(steps)

# Plot Epsilon-Greedy results
plt.figure(figsize=(12, 6))
for eps, (_, rewards) in zip(epsilon_values, eps_results):
    plt.plot(rewards, label=f'Epsilon-Greedy (ε={eps})')
plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')
plt.legend()
plt.title('Epsilon-Greedy Performance with Different ε Values')
plt.savefig("figures/epsilon_greedy_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot UCB results
plt.figure(figsize=(12, 6))
for c, (_, rewards) in zip(c_values, ucb_results):
    plt.plot(rewards, label=f'UCB (c={c})')
plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')
plt.legend()
plt.title('UCB Performance with Different c Values')
plt.savefig("figures/ucb_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot Thompson Sampling results
plt.figure(figsize=(12, 6))
plt.plot(thompson_result[1], label='Thompson Sampling', linestyle='--')
plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')
plt.legend()
plt.title('Thompson Sampling Performance')
plt.savefig("figures/thompson_sampling_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Overall Comparison
plt.figure(figsize=(12, 6))
for eps, (_, rewards) in zip(epsilon_values, eps_results):
    plt.plot(rewards, label=f'Epsilon-Greedy (ε={eps})')
for c, (_, rewards) in zip(c_values, ucb_results):
    plt.plot(rewards, label=f'UCB (c={c})')
plt.plot(thompson_result[1], label='Thompson Sampling', linestyle='--')
plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')
plt.legend()
plt.title('Overall Comparison of Multi-Armed Bandit Algorithms')
plt.savefig("figures/multi_armed_bandit_overall_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

