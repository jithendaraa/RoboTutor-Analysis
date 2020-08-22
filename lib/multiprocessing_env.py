import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action = data[0]
            max_timestep = data[1]
            timestep = data[2]
            activityName = data[3]
            bayesian_update = data[4]
            plot = data[5]
            prints = data[6]
            next_state, reward, student_response, done, posterior = env.step(action, max_timestep, timesteps=timestep, activityName=activityName, bayesian_update=bayesian_update, plot=plot, prints=prints)
            if done:
                next_state = env.reset()
            remote.send((next_state, reward, student_response, done, posterior))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'checkpoint':
            ob = env.checkpoint()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        print("RES")
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions, max_timesteps, timesteps=None, activityNames=None, bayesian_updates=True, plots=False, printss=False):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, student_responses, dones, posteriors):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - student_responses: an array of student performances on activities
         - dones: an array of "episode done" booleans
         - posteriors:  a list of posterior knowledge
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions, max_timesteps, timesteps=None, activityNames=None, bayesian_updates=True, plots=False, printss=False):
        self.step_async(actions, max_timesteps, timesteps=None, activityNames=None, bayesian_updates=True, plots=False, printss=False)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for process in self.ps:
            process.daemon = True # if the main process crashes, we should not cause things to hang
            process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions, max_timesteps, timesteps=None, activityNames=None, bayesian_updates=True, plots=False, printss=False):
        
        if timesteps == None:
            timesteps = [None] * len(actions)
        if activityNames == None:
            activityNames = [None] * len(actions)
        
        for remote, action, max_timestep, timestep, activityName in zip(self.remotes, actions, max_timesteps, timesteps, activityNames):
            data = [action, max_timestep, timestep, activityName, bayesian_updates, plots, printss]
            remote.send(('step', data))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        next_states, rewards, student_responses, dones, posteriors = zip(*results)
        return np.stack(next_states), np.stack(rewards), np.stack(student_responses), np.stack(dones), np.stack(posteriors)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def checkpoint(self):
        for remote in self.remotes:
            remote.send(('checkpoint', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs