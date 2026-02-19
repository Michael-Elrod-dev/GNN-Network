import torch
import numpy as np
from enum import IntEnum

TILE_PIXELS = 32

COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'orange': np.array([255, 165, 0]),
    'cyan': np.array([0, 255, 255]),
    'magenta': np.array([255, 0, 255]),
    'lime': np.array([0, 255, 0]),
    'pink': np.array([255, 192, 203]),
    'teal': np.array([0, 128, 128]),
    'lavender': np.array([230, 230, 250]),
    'brown': np.array([165, 42, 42]),
    'beige': np.array([245, 245, 220]),
    'maroon': np.array([128, 0, 0]),
    'mint': np.array([192, 255, 192]),
    'olive': np.array([128, 128, 0]),
    'coral': np.array([255, 127, 80]),
    'navy': np.array([0, 0, 128]),
    'white': np.array([255, 255, 255]),
    'black': np.array([0, 0, 0]),
}
COLOR_NAMES = sorted(list(COLORS.keys()))

COLOR_TO_IDX = {
    'red': 1,
    'green': 2,
    'blue': 3,
    'purple': 4,
    'yellow': 5,
    'grey': 6,
    'orange': 7,
    'cyan': 8,
    'magenta': 9,
    'lime': 10,
    'pink': 11,
    'teal': 12,
    'lavender': 13,
    'brown': 14,
    'beige': 15,
    'maroon': 16,
    'mint': 17,
    'olive': 18,
    'coral': 19,
    'navy': 20,
    'white': 21,
    'black': 22,
}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'agent': 3,
    'goal': 4,
    'obstacle': 5,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

DIR_TO_VEC = [
    np.array((1, 0)),
    np.array((0, 1)),
    np.array((-1, 0)),
    np.array((0, -1)),
]

class Actions(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3

def print_info(id, obs, node_obs, adj, action=None, reward=None, next_obs=None, next_node_obs=None, next_adj=None, done=None):
    print(f'-' * 40)
    if done is not None:
        print(f'Agent: {id} | Action: {action} | Reward: {reward} | Done: {done}')
    else:
        print(f'Reset:\nAgent: {id}')
    
    print(f'\nObservation: {obs.shape} {type(obs)}')
    print(obs)

    if done is not None:
        print(f'\nNext Observation: {next_obs.shape} {type(next_obs)}')
        print(next_obs)

    print(f'\nNode Observation: {node_obs.shape} {type(node_obs)}')
    print(node_obs)

    if done is not None:
        print(f'\nNext Node Observation: {next_node_obs.shape} {type(next_node_obs)}')
        print(next_node_obs)

    print(f'\nAdjacency Matrix: {adj.shape} {type(adj)}')
    print(adj)

    if done is not None:
        print(f'\nNext Adjacency Matrix: {next_adj.shape} {type(next_adj)}')
        print(next_adj)
    print(f'-' * 40)

def unbind(agent_id, obs, node_obs, adj, action=None, reward=None, next_obs=None, next_node_obs=None, next_adj=None, dones=None):
    agent_id = torch.unbind(agent_id)
    obs = torch.unbind(obs)
    node_obs = torch.unbind(node_obs)
    adj = torch.unbind(adj)
    if action is not None:
        action = torch.unbind(action)
        reward = torch.unbind(reward)
        next_obs = torch.unbind(next_obs)
        next_node_obs = torch.unbind(next_node_obs)
        next_adj = torch.unbind(next_adj)
        dones = torch.unbind(dones)
        return agent_id, obs, node_obs, adj, action, reward, next_obs, next_node_obs, next_adj, dones
    return agent_id, obs, node_obs, adj