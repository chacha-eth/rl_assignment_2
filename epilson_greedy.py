import numpy as np

class EpsilonGreedy:
    def __init__(self, num_arms, true_rewards, epsilon):
        self.num_arms = num_arms
        self.true_rewards = true_rewards
        self.epsilon = epsilon
        self.q_values = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

    def run(self, steps):
        total_reward = 0
        rewards = []
        for _ in range(steps):
            arm = self.select_arm()
            reward = np.random.normal(self.true_rewards[arm], 1)
            self.update(arm, reward)
            total_reward += reward
            rewards.append(total_reward / (_ + 1))
        return total_reward, rewards