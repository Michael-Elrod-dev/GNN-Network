import numpy as np
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    closed = False
    viewer = None
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


def graphworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    ag_id = None

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            ob, node_ob, adj, reward, done, info = env.step(data)
            if ag_id is None:
                n_agents = ob.shape[0]
                ag_id = np.arange(n_agents, dtype=np.int32).reshape(-1, 1)
            remote.send((ob, ag_id, node_ob, adj, reward, done, info))

        elif cmd == "reset":
            ob, node_ob, adj = env.reset()
            n_agents = ob.shape[0]
            ag_id = np.arange(n_agents, dtype=np.int32).reshape(-1, 1)
            remote.send((ob, ag_id, node_ob, adj))

        elif cmd == "close":
            env.close()
            remote.close()
            break

        elif cmd == "get_spaces":
            remote.send(
                (
                    env.observation_space,
                    env.share_observation_space,
                    env.action_space,
                    env.node_observation_space,
                    env.adj_observation_space,
                    env.edge_observation_space,
                    env.agent_id_observation_space,
                    env.share_agent_id_observation_space,
                )
            )

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class GraphSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=graphworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                daemon=True,
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        (
            observation_space,
            share_observation_space,
            action_space,
            node_observation_space,
            adj_observation_space,
            edge_observation_space,
            agent_id_observation_space,
            share_agent_id_observation_space,
        ) = self.remotes[0].recv()

        ShareVecEnv.__init__(self, nenvs, observation_space, share_observation_space, action_space)
        self.node_observation_space = node_observation_space
        self.adj_observation_space = adj_observation_space
        self.edge_observation_space = edge_observation_space
        self.agent_id_observation_space = agent_id_observation_space
        self.share_agent_id_observation_space = share_agent_id_observation_space

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, ag_ids, node_obs, adj, rews, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(ag_ids),
            np.stack(node_obs),
            np.stack(adj),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, ag_ids, node_obs, adj = zip(*results)
        return np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


class GraphDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.n_agents = env.observation_space.shape[0] if hasattr(env.observation_space, "n") else len(env.agents)
        self.node_observation_space = env.node_observation_space
        self.adj_observation_space = env.adj_observation_space
        self.edge_observation_space = env.edge_observation_space
        self.agent_id_observation_space = env.agent_id_observation_space
        self.share_agent_id_observation_space = env.share_agent_id_observation_space
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs_list, node_obs_list, adj_list, rews_list, dones_list, infos_list = zip(*results)
        obs = np.stack(obs_list)
        node_obs = np.stack(node_obs_list)
        adj = np.stack(adj_list)
        rews = np.stack(rews_list)
        dones = np.stack(dones_list)

        n_agents = obs.shape[1]
        ag_ids = np.tile(
            np.arange(n_agents, dtype=np.int32).reshape(1, -1, 1),
            (len(self.envs), 1, 1),
        )

        self.actions = None
        return obs, ag_ids, node_obs, adj, rews, dones, infos_list

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs_list, node_obs_list, adj_list = zip(*results)
        obs = np.stack(obs_list)
        node_obs = np.stack(node_obs_list)
        adj = np.stack(adj_list)

        n_agents = obs.shape[1]
        ag_ids = np.tile(
            np.arange(n_agents, dtype=np.int32).reshape(1, -1, 1),
            (len(self.envs), 1, 1),
        )
        return obs, ag_ids, node_obs, adj

    def close(self):
        for env in self.envs:
            env.close()
