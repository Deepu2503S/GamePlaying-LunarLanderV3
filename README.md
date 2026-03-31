**🚀 Lunar Lander AI Agent (PSO-Based)**

This project implements an AI agent for solving the LunarLander-v3 environment using Particle Swarm Optimization (PSO) instead of traditional RL methods like DQN or Policy Gradients.

It also includes tools to:

Train an agent
Evaluate performance
Play the game manually

**🧠 Approach**

Instead of learning via gradients, this project uses Particle Swarm Optimization (PSO):

Each particle = a candidate policy (weights)
Policies are evaluated using environment rewards
Best policies guide the swarm
Final output = best-performing policy

**
⚙️ Installation**

Make sure you have Python 3.8+ installed.

pip install gymnasium[box2d] numpy pygame
🏋️ Training the Agent

**Run:**

python train_agent2113.py
Uses PSO with:
200 particles
50 generations
Saves best policy as:
best_policy_2113.npy

**📌 Training logic includes reward shaping:**

Bonus for stable landing
Penalty for bad angles
Encouragement for low velocity


**🎮 Policy Representation**

The policy is a simple linear model:

action = argmax(observation × W)
Observation: 8-dimensional
Actions: 4 discrete actions
Weights: stored as a flattened vector


**📊 Evaluating the Agent**

**Run:**

python evaluate_agent.py --filename best_policy_2113.npy --policy_module policy_2113
Runs 100 episodes
First 5 episodes are rendered
Outputs average reward

**Evaluation script:**

🎮 Play the Game Yourself

**Run:**

python play_lunar_lander.py

Controls:
**Key	Action**
W	Main engine
A	Left engine
D	Right engine
S	Do nothing



**📈 Features**

✅ PSO-based RL (non-gradient approach)
✅ Custom reward shaping
✅ Modular design (train / eval / play)
✅ Lightweight policy (no neural networks)
✅ Easy to extend


**🧪 Future Improvements**

Replace linear policy with neural networks

Try PPO / DQN for comparison

Hyperparameter tuning (particles, generations)

Visualization of training progress

Logging with TensorBoard


**🙌 Acknowledgements**

OpenAI Gym / Gymnasium
Box2D physics engine

**👨‍💻 Author**
Deepanshu Singh
B.Tech CSE
IIIT Guwahati
