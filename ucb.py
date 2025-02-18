import numpy as np

class UCB:
    def __init__(self, num_arms, true_rewards, c):
        self.num_arms = num_arms
        self.true_rewards = true_rewards
        self.c = c
        self.q_values = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        if 0 in self.arm_counts:
            return np.argmin(self.arm_counts)
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.arm_counts)
        return np.argmax(ucb_values)

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
