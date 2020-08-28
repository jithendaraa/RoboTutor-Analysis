import sys
import os
sys.path.append(os.getcwd()+'/lib')
sys.path.append(os.getcwd()+'/../')

import argparse
import math
import random
import numpy as np
from pathlib import Path
import pickle

import torch 
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from common import mkdir
from multiprocessing_env import SubprocVecEnv

from actor_critic_agent import ActorCritic
from environment import StudentEnv
from student_simulator import StudentSimulator

from helper import *
from reader import *
from ppo_helper import *

CONSTANTS = {
                'STUDENT_MODEL_NAME': 'hotDINA_full',
                'VILLAGE'           : '130',
                'NUM_OBS'           : '100',
                "STUDENT_ID"        : "new_student",
                "ENV_ID"            : "RoboTutor",
                "NUM_ENVS"          : 4,
                "TARGET_P_KNOW"     : 0.90,
                "STATE_SIZE"        : 22,
                "ACTION_SIZE"       : 43,
                "FC1_DIMS"          : 256,
                "RUN_NUM"           : 0,
                "AVG_OVER_RUNS"     : 10,
                "MAX_TIMESTEPS"     : 50,
                "LEARNING_RATE"     : 1e-4,
                "GAMMA"             : 0.99,
                "GAE_LAMBDA"        : 0.95,     # smoothing factor
                "PPO_EPSILON"       : 0.2,      # clip betweeen 0.8 and 1.2
                "CRITIC_DISCOUNT"   : 0.5,      # critic loss usually greater than actor loss, so we scale it down by this value; improves training
                "ENTROPY_BETA"      : 0.001,    # amount of importance given to entropy which helps to explore
                "PPO_STEPS"         : 10,      # number of transitions we sample for each training iteration; Each step collects transition from parallel envs. So total data is PPO_STEPS 8 NUM_ENVS -> 256 * 8 = 2048
                "MINI_BATCH_SIZE"   : 32,       # number of samples that are randomly selected from the total amount of stored data
                "PPO_EPOCHS"        : 5,
                "TEST_EPOCHS"       : 5,
                "NUM_TESTS"         : 50,
                'PATH_TO_ACTIIVTY_TABLE': 'Data/Activity_table_v4.1_22Apr2020.pkl'
    }

def set_constants(args):
    CONSTANTS['NUM_OBS']            = args.observations
    CONSTANTS['VILLAGE']            = args.village_num
    CONSTANTS['STUDENT_ID']         = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['ENV_ID']             = args.student_model_name

def get_data_dict(uniq_student_ids, kc_list, path=''):
    if CONSTANTS['STUDENT_MODEL_NAME'] == 'ActivityBKT':
        data_dict = extract_activity_table(path + CONSTANTS['PATH_TO_ACTIVITY_TABLE'], uniq_student_ids, kc_list, CONSTANTS["NUM_OBS"], CONSTANTS['STUDENT_ID'])
    
    elif CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_skill' or CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_full':
        path_to_data_file = os.getcwd() + '/' + path +'../hotDINA/pickles/data/data'+ CONSTANTS['VILLAGE'] + '_' + CONSTANTS['NUM_OBS'] +'.pickle'
        data_file = Path(path_to_data_file)
        if data_file.is_file() == False:
            # if data_file does not exist, get it
            os.chdir(path + '../hotDINA')
            get_data_file_command = 'python get_data_for_village_n.py -v ' + CONSTANTS['VILLAGE'] + ' -o ' + CONSTANTS['NUM_OBS'] 
            os.system(get_data_file_command)
            os.chdir('../RoboTutor-Analysis')

        os.chdir(path + '../hotDINA')
        with open(path_to_data_file, 'rb') as handle:
            data_dict = pickle.load(handle)
        os.chdir('../RoboTutor-Analysis')
    return data_dict
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=CONSTANTS["ENV_ID"], help="Name of the run")
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument('-smn', '--student_model_name', default=CONSTANTS['STUDENT_MODEL_NAME'], help="Student model name")
    args = parser.parse_args()
    set_constants(args)

    path = '../'
    kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data(path)

    student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"],
                                        path=path)
    
    uniq_student_ids = student_simulator.uniq_student_ids
    kc_list = student_simulator.kc_list
    cta_df = student_simulator.cta_df
    
    if CONSTANTS['STUDENT_ID'] not in uniq_student_ids:
        CONSTANTS['STUDENT_ID'] = uniq_student_ids[0]

    student_num = uniq_student_ids.index(CONSTANTS["STUDENT_ID"])
    CONSTANTS['STATE_SIZE'] = len(kc_list)
    
    data_dict = get_data_dict(uniq_student_ids, kc_list, path)
    train_students = []
    for s_id in uniq_student_ids:
        if s_id != args.student_id:
            train_students.append(uniq_student_ids.index(s_id))
    student_simulator.update_on_log_data(data_dict, plot=False, bayesian_update=False, train_students=train_students)
    
    env = StudentEnv(student_simulator=student_simulator,
                    skill_groups=uniq_skill_groups,
                    skill_group_to_activity_map = skill_group_to_activity_map,
                    action_size=CONSTANTS["ACTION_SIZE"],
                    student_id=CONSTANTS["STUDENT_ID"])
    env.checkpoint()

    init_p_know                   = env.reset()
    init_avg_p_know               = np.mean(np.array(init_p_know))
    target_avg_p_know             = CONSTANTS["TARGET_P_KNOW"]
    final_p_know                  = init_p_know.copy()
    final_avg_p_know              = init_avg_p_know
    CONSTANTS["TARGET_REWARD"]    = 1000 * (target_avg_p_know - init_avg_p_know)

    # clear log files: ppo_logs
    clear_files("ppo", True, path='RL_agents/')
    mkdir('.', 'checkpoints')
    writer = SummaryWriter(comment="ppo_" + args.name)

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    print('Device:', device)

    # Prepare environments
    envs = [make_env(i+1, 
                    student_simulator, 
                    CONSTANTS["STUDENT_ID"], 
                    uniq_skill_groups, 
                    skill_group_to_activity_map, 
                    CONSTANTS["ACTION_SIZE"]) for i in range(CONSTANTS["NUM_ENVS"])]
    envs = SubprocVecEnv(envs)
    envs.checkpoint()
    
    num_inputs  = CONSTANTS["STATE_SIZE"]
    n_actions = CONSTANTS["ACTION_SIZE"]
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[num_inputs], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=n_actions)
    # model.load_state_dict(torch.load("checkpoints/RoboTutor_best_+804.453_30720.dat"))
    frame_idx = 0
    train_epoch = 0
    best_reward = None
    state       = envs.reset()
    early_stop  = False

    while not early_stop:
        # lists to store training data
        log_probs       = []
        critic_values   = []
        states          = []
        actions         = []
        rewards         = []
        dones           = []
        timesteps       = 0

        for _ in range(CONSTANTS["PPO_STEPS"]):
            timesteps += 1
            if isinstance(state, np.ndarray) == False and state.get_device() != 'cpu:0':
                state = state.to('cpu:0')
            state = torch.FloatTensor(state)
            if state.get_device() != 'cuda:0':
                state = state.to(device)
            
            policy, critic_value = model.forward(state)
            policy = F.softmax(policy, dim=1)   # softmax ensures actions add up to one which is a requirement for probabilities
            action_probs = torch.distributions.Categorical(policy)
            action = action_probs.sample()
            activity_names = []
            
            for item in action.tolist():
                activity_name = np.random.choice(skill_group_to_activity_map[str(item)])
                activity_names.append(activity_name)
            
            next_state, reward, student_response, done, posterior = envs.step(action.cpu().numpy(), [timesteps] * CONSTANTS["NUM_ENVS"], [CONSTANTS["MAX_TIMESTEPS"]] * CONSTANTS["NUM_ENVS"], activity_names)
            log_prob = action_probs.log_prob(action)
            log_probs.append(log_prob)
            critic_values.append(critic_value)
            rewards.append(torch.Tensor(reward).unsqueeze(1).to(device))
            dones.append(torch.Tensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            actions.append(action)
            state = next_state.copy()
            frame_idx += 1

        _, critic_value_ = model(torch.Tensor(next_state).to(device))
        returns = compute_gae(critic_value_, rewards, dones, critic_values, CONSTANTS["GAMMA"], CONSTANTS["GAE_LAMBDA"])
        returns         = torch.cat(returns).detach()
        log_probs       = torch.cat(log_probs).detach()
        critic_values   = torch.cat(critic_values).detach()
        states          = torch.cat(states)
        actions         = torch.cat(actions)
        advantage       = returns - critic_values
        advantage       = normalize(advantage)

        # According to PPO paper: (states, actions, log_probs, returns, advantage) is together referred to as a "trajectory"
        ppo_update(model, frame_idx, states, actions, log_probs, returns, advantage, CONSTANTS)
        train_epoch += 1
        break
        
        # if train_epoch % CONSTANTS["TEST_EPOCHS"] == 0:
        #     student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
        #                                 observations=CONSTANTS["NUM_OBS"], 
        #                                 student_model_name=CONSTANTS["STUDENT_MODEL_NAME"])
        #     data_dict = get_data_dict(uniq_student_ids, kc_list)
        #     # student_simulator.update_on_log_data(data_dict, plot=False, bayesian_update=True)
        #     env = StudentEnv(student_simulator=student_simulator,
        #             skill_groups=uniq_skill_groups,
        #             skill_group_to_activity_map = skill_group_to_activity_map,
        #             action_size=CONSTANTS["ACTION_SIZE"],
        #             student_id=CONSTANTS["STUDENT_ID"])
        #     env.checkpoint()
        #     test_reward = np.mean([test_env(env, model, device, CONSTANTS, skill_group_to_activity_map, uniq_skill_groups) for _ in range(CONSTANTS["NUM_TESTS"])])
        #     writer.add_scalar("test_rewards", test_reward, frame_idx)
        #     print('Frame %s. reward: %s' % (frame_idx, test_reward))
        #     print("FINAL AVG P(Know) after run ", CONSTANTS["RUN_NUM"], ": ", (test_reward/1000 + init_avg_p_know))
        #     print("FINAL P(Know): after run ", CONSTANTS["RUN_NUM"], ": ", env.state)
        #     final_p_know = env.state
        #     final_avg_p_know = np.mean(np.array(final_p_know))
        #     # Save a checkpoint every time we achieve a best reward
        #     if best_reward is None or best_reward < test_reward:
        #         if best_reward is not None:
        #             print("Best reward updated: %.3f -> %.3f Target reward: %.3f" % (best_reward, test_reward, CONSTANTS["TARGET_REWARD"]))
        #             name = CONSTANTS['STUDENT_MODEL_NAME'] + ("best_%+.3f_%d.dat" % (test_reward, frame_idx))
        #             fname = os.path.join('.', 'checkpoints', name)
        #             torch.save(model.state_dict(), fname)
        #         best_reward = test_reward
        #     if test_reward > CONSTANTS["TARGET_REWARD"]: 
        #         early_stop = True

    # print("INIT P(Know): \n", init_p_know)
    # print("FINAL P(Know): \n", final_p_know)
    # print("IMPROVEMENT PER SKILL: \n", np.array(final_p_know) - np.array(init_p_know))
    # print("INIT AVG P(KNOW): ", init_avg_p_know)
    # print("FINAL AVG P(KNOW): ", final_avg_p_know)
    # print("TOTAL RUNS: ", CONSTANTS["RUN_NUM"])