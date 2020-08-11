import sys

import numpy as np
import torch
import torch.nn.functional as F

from environment import StudentEnv
from student_simulator import StudentSimulator

from helper import *
from reader import *

STUDENT_ID = ["CQCKBY_105"]

def make_env(i, student_simulator, student_id, uniq_skill_groups, skill_group_to_activity_map, ACTION_SIZE):
    # returns a functions which creates a single environment
    def _thunk():
        env = StudentEnv(student_simulator=student_simulator,
                            skill_groups=uniq_skill_groups,
                            skill_group_to_activity_map = skill_group_to_activity_map,
                            action_size=ACTION_SIZE,
                            student_id=student_id,
                            env_num=i)
        return env
    return _thunk

def compute_gae(critic_value_, rewards, dones, critic_values, gamma, lam):
    # The arguments of this function contain data across each parallel environment; from most recent experience to earlier
    critic_values = critic_values + [critic_value_]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        # delta = bellman_equation - V(s); basically the advantage
        delta = rewards[step] + (gamma * critic_values[step + 1] * dones[step]) - critic_values[step]
        gae = delta + gamma * lam * dones[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + critic_values[step])
    # returns is a list PPO_STEPS long and NUM_ENVS wide
    return returns

def ppo_iter(states, actions, log_probs, returns, advantages, PPO_STEPS, NUM_ENVS, MINI_BATCH_SIZE):
    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions.view(PPO_STEPS * NUM_ENVS, -1)[rand_ids, :], log_probs.view(PPO_STEPS * NUM_ENVS, -1)[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

def test_env(env, model, device, CONSTANTS, skill_group_to_activity_map, uniq_skill_groups, deterministic=False):
    
    state = env.reset()
    init_state = state.copy()
    done = False
    total_reward = 0
    timesteps = 0
    while not done:
        timesteps += 1
        state = torch.Tensor(state).unsqueeze(0).to(device)
        policy, _ = model(state)
        policy = F.softmax(policy, dim=1)
        action_probs = torch.distributions.Categorical(policy)
        action = torch.max(policy.view(-1), 0)[1].item()
        
        if deterministic is False:
            action = action_probs.sample().item()
        
        activity_name = np.random.choice(skill_group_to_activity_map[str(action)])

        next_state, reward, student_response, done, posterior = env.step(action, timesteps, CONSTANTS["MAX_TIMESTEPS"], activity_name)

        skill_group = uniq_skill_groups[action]
        prior = state[0]
        posterior = torch.Tensor(next_state).unsqueeze(0).to(device)[0]
        posterior_avg_know  =  torch.mean(posterior).item()
        prior_avg_know      =  torch.mean(state).item()
        gain = (posterior_avg_know - prior_avg_know)

        prior_know = []
        posterior_know = []

        for skill_idx in skill_group:
            prior_know.append(prior[skill_idx].item())
            posterior_know.append(posterior[skill_idx].item())

        run_text = "Run Number: " + str(CONSTANTS["RUN_NUM"]) + "\Action: " + str(action) + " Skill group: " + str(skill_group) + "\nPrior Know: " + str(prior_know) + "\nPost. Know: " + str(posterior_know) + "\nTimestep: " + str(timesteps) + " Reward: " + str(reward) + " ActivityName: " + activity_name + "\nPrior: " + str(prior_avg_know) + " Posterior: " + str(posterior_avg_know) + "\nGain: " + str(gain) + "\n_____________________________________________________________________________\n"

        state = next_state
        total_reward += reward
    
        with open("ppo_logs/test_run.txt", "a") as f:
            f.write(run_text)

    with open("ppo_logs/rewards.txt", "a") as f:
        if CONSTANTS["RUN_NUM"] == 0:
            f.write("0,"+ str(np.mean(np.array(init_state))) + "\n")

        text = str(CONSTANTS["RUN_NUM"]) + "," + str(posterior_avg_know) + "\n"
        f.write(text)
    
    if CONSTANTS["RUN_NUM"] % CONSTANTS["AVG_OVER_RUNS"] == 0:
        with open("ppo_logs/avg_scores.txt", "a") as f:
            text = str(CONSTANTS["RUN_NUM"]/CONSTANTS["AVG_OVER_RUNS"]) + "," + str(posterior_avg_know) + "\n"
            f.write(text)

    CONSTANTS["RUN_NUM"] += 1

    with open("ppo_logs/test_run.txt", "a") as f:
        f.write("Total Reward: " + str(total_reward) + "\n")

    return total_reward

def ppo_update(model, frame_idx, states, actions, log_probs, returns, advantages, CONSTANTS):
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0
    clip_param = CONSTANTS["PPO_EPSILON"]


    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(CONSTANTS["PPO_EPOCHS"]):
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages, CONSTANTS["PPO_STEPS"], CONSTANTS["NUM_ENVS"], CONSTANTS["MINI_BATCH_SIZE"]):
            policy, critic_value = model(state)
            policy = F.softmax(policy, dim=1)
            action_probs = torch.distributions.Categorical(policy)
            entropy = action_probs.entropy().mean()
            new_log_probs = action_probs.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - critic_value).pow(2).mean()

            loss = CONSTANTS["CRITIC_DISCOUNT"] * critic_loss + actor_loss - CONSTANTS["ENTROPY_BETA"] * entropy

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
            
            count_steps += 1

        # writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
        # writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
        # writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
        # writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
        # writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
        # writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)

def play_env(env, model, device, CONSTANTS, deterministic=True):
    state = env.reset()
    init_state = state.copy()
    done = False
    total_reward = 0
    timesteps = 0
    prior = None
    posterior = None

    while not done:
        timesteps += 1
        state = torch.Tensor(state).unsqueeze(0).to(device)
        policy, _ = model(state)
        policy = F.softmax(policy, dim=1)
        action_probs = torch.distributions.Categorical(policy)
        action = torch.max(policy.view(-1), 0)[1].item()
        
        if deterministic is False:
            action = action_probs.sample().item()

        next_state, reward, student_response, done, posterior = env.step(action, timesteps, CONSTANTS["MAX_TIMESTEPS"])

        activityName = np.random.choice(skill_group_to_activity_map[str(action)])
        skill_group = uniq_skill_groups[action]
        prior = state[0]
        posterior = torch.Tensor(next_state).unsqueeze(0).to(device)[0]
        posterior_avg_know  =  torch.mean(posterior).item()
        prior_avg_know      =  torch.mean(state).item()
        gain = (posterior_avg_know - prior_avg_know)

        prior_know = []
        posterior_know = []

        for skill_idx in skill_group:
            prior_know.append(prior[skill_idx].item())
            posterior_know.append(posterior[skill_idx].item())

        run_text = "Run Number: " + str(timesteps) + "\Action: " + str(action) + " Skill group: " + str(skill_group) + "\nPrior Know: " + str(prior_know) + "\nPost. Know: " + str(posterior_know) + "\nTimestep: " + str(timesteps) + " Reward: " + str(reward) + " ActivityName: " + activityName + "\nPrior: " + str(prior_avg_know) + " Posterior: " + str(posterior_avg_know) + "\nGain: " + str(gain) + "\n_____________________________________________________________________________\n"
        print(run_text)

        state = next_state
        total_reward += reward
    
        with open("ppo_logs/play_run.txt", "a") as f:
            f.write(run_text)

    with open("ppo_logs/play_run.txt", "a") as f:
        f.write("Total Reward: " + str(total_reward) + "\n")
    
    print("In %d steps we got %.3f reward" % (timesteps, total_reward))
    learning_progress = env.activity_bkt.activity_learning_progress

    return total_reward, posterior, learning_progress   

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

