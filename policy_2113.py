import numpy as np

def policy_action(params, observation):
    W = params[:32].reshape(8, 4)
    logits = np.dot(observation, W)
    return np.argmax(logits)