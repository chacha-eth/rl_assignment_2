import numpy as np
class ThompsonSampling:
    def __init__(self, num_arms, true_probs):
        self.num_arms = num_arms
        self.true_probs = true_probs
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def run(self, steps):
        total_reward = 0
        rewards = []
        for _ in range(steps):
            arm = self.select_arm()
            reward = np.random.binomial(1, self.true_probs[arm])
            self.update(arm, reward)
            total_reward += reward
            rewards.append(total_reward / (_ + 1))
        return total_reward, rewards