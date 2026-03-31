# 🚀 Lunar Lander AI Agent (PSO-Based)

This project implements an AI agent for solving the **LunarLander-v3** environment using **Particle Swarm Optimization (PSO)** instead of traditional reinforcement learning methods.

## 📂 Project Structure

```
.
├── train_agent2113.py
├── policy_2113.py
├── evaluate_agent.py
├── play_lunar_lander.py
├── best_policy_2113.npy
```

## 🧠 Approach

- Uses Particle Swarm Optimization (PSO)
- Each particle represents a policy (weights)
- Best policies guide the swarm

Policy:
```
action = argmax(observation × W)
```

## ⚙️ Installation

```
pip install gymnasium[box2d] numpy pygame
```

## 🏋️ Training

```
python train_agent2113.py
```

## 📊 Evaluation

```
python evaluate_agent.py --filename best_policy_2113.npy --policy_module policy_2113
```

## 🎮 Play

```
python play_lunar_lander.py
```

Controls:
- W → Main engine  
- A → Left engine  
- D → Right engine  
- S → Do nothing  

## 👨‍💻 Author

Deepanshu Singh  
B.Tech CSE '27  
IIIT Guwahati
