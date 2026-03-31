import gymnasium as gym
import numpy as np
import math

env = gym.make("LunarLander-v3")

num_particles = 200
num_generations = 50
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
w_initial = 0.9
c1_initial = 2.7
c2_initial = 1.4

try:
    gbest_position = np.load("best_policy_2113.npy")
    particles = np.random.uniform(-1, 1, (num_particles, state_dim * action_dim))
    particles[0] = gbest_position
    print("Loaded previous best policy for continued training.")
except FileNotFoundError:
    particles = np.random.uniform(-1, 1, (num_particles, state_dim * action_dim))
    print("No previous best policy found. Starting from scratch.")

velocities = np.random.uniform(-0.5, 0.5, (num_particles, state_dim * action_dim))
pbest_positions = particles.copy()
pbest_scores = np.full(num_particles, -float("inf"))
gbest_score = -float("inf")

def policy_action(params, state):
    W = params.reshape(state_dim, action_dim)
    return np.argmax(np.dot(state, W))

def evaluate_policy(params, episodes=5):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        episodic_reward = 0.0
        done = False

        while not done:
            action = policy_action(params, state)
            state, reward, done, truncated, _ = env.step(action)
            x, y, vx, vy, angle, angular_velocity, leftl, rightl = state
            if abs(x)<0.4:
                reward+=10
            if abs(angle)>0.2:
                reward-=10
            if abs(vy) < 0.3:
                reward+=10
            if abs(vx) < 0.3:
                reward+=10
            if leftl and rightl :
                reward+=50
            elif leftl or rightl :
                reward+=20
            
            episodic_reward += reward

            if truncated:
                break

        total_reward += episodic_reward

    return total_reward / episodes

for gen in range(num_generations):
    generation_rewards = []
    
    w = max(0.3, w_initial - (gen / num_generations) * 0.3)
    c1 = c1_initial - (gen / num_generations) * 1.5  
    c2 = c2_initial + (gen / num_generations) * 1.5
    random_exploration = max(0.15, 0.3 - (gen / num_generations) * 0.1)

    for j in range(num_particles):
        score = evaluate_policy(particles[j])
        generation_rewards.append(score)

        if score > pbest_scores[j]:
            pbest_scores[j] = score
            pbest_positions[j] = particles[j].copy()
        if score > gbest_score:
            gbest_score = score
            gbest_position = particles[j].copy()

    for i in range(num_particles):
        inertia = w * velocities[i]
        cognitive = c1 * np.random.rand() * (pbest_positions[i] - particles[i])
        social = c2 * np.random.rand() * (gbest_position - particles[i])

        velocities[i] = inertia + cognitive + social + random_exploration * np.random.randn(state_dim * action_dim)
        particles[i] += velocities[i]

        if np.random.rand() < 0.1:
            particles[i] = np.random.uniform(-1, 1, state_dim * action_dim)

    print(f"Generation {gen+1}: Best Reward = {gbest_score:.2f}", flush=True)

np.save("best_policy_2113.npy", gbest_position)
print(f"Training completed. Best Reward: {gbest_score:.2f}")