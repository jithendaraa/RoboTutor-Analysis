import sys
sys.path.append("../lib")
sys.path.append('../')

import argparse
import math
import os
import random
import numpy as np

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

train_and_play = False

CONSTANTS = {
                "NUM_ENVS"          : 8,
                "STUDENT_ID"        : "new_student",
                "ENV_ID"            : "RoboTutor",
                "TARGET_P_KNOW"     : 0.85,
                "STATE_SIZE"        : 18,
                "ACTION_SIZE"       : 33,
                "FC1_DIMS"          : 256,
                "RUN_NUM"           : 0,
                "AVG_OVER_RUNS"     : 50,
                "MAX_TIMESTEPS"     : 250,
                "LEARNING_RATE"     : 1e-4,
                "GAMMA"             : 0.99,
                "GAE_LAMBDA"        : 0.95,     # smoothing factor
                "PPO_EPSILON"       : 0.2,      # clip betweeen 0.8 and 1.2
                "CRITIC_DISCOUNT"   : 0.5,      # critic loss usually greater than actor loss, so we scale it down by this value; improves training
                "ENTROPY_BETA"      : 0.001,    # amount of importance given to entropy which helps to explore
                "PPO_STEPS"         : 256,      # number of transitions we sample for each training iteration; Each step collects transition from parallel envs. So total data is PPO_STEPS 8 NUM_ENVS -> 256 * 8 = 2048
                "MINI_BATCH_SIZE"   : 64,       # number of samples that are randomly selected from the total amount of stored data
                "PPO_EPOCHS"        : 10,
                "TEST_EPOCHS"       : 10,
                "NUM_TESTS"         : 50,
                'STUDENT_MODEL_NAME': 'ActivityBKT',
                'VILLAGE'           : '130',
                'NUM_OBS'           : '1000',
                'MATRIX_TYPE'       : 'math',
            }

def set_constants(args):
    CONSTANTS['ENV_ID']     = args.name
    CONSTANTS['NUM_OBS']    = args.observations
    CONSTANTS['VILLAGE']    = args.village_num
    CONSTANTS['MATRIX_TYPE']= args.matrix_type

# TRAIN
if __name__ == "__main__":
    # from ppo_helper import normalize, compute_gae, test_env, ppo_update, make_env

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=CONSTANTS["ENV_ID"], help="Name of the run")
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument("-m", "--matrix_type", default=CONSTANTS["MATRIX_TYPE"], help="Matrix type for the 3 content matrices or 'all'")
    args = parser.parse_args()

    set_constants(args)

    os.chdir('..')
    kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()

    student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"],
                                        matrix_type=CONSTANTS['MATRIX_TYPE'])
    
    uniq_student_ids = student_simulator.uniq_student_ids
    data_dict = extract_activity_table(uniq_student_ids, kc_list, CONSTANTS['MATRIX_TYPE'], CONSTANTS["NUM_OBS"])
    student_simulator.update_on_log_data(data_dict)


    # activity_bkt, activity_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(1.0, train_students=[CONSTANTS["STUDENT_ID"]])

    # student_num         = uniq_student_ids.index(CONSTANTS["STUDENT_ID"])
    # initial_state       = np.array(activity_bkt.know[student_id])
    # env                 = StudentEnv(initial_state, 
    #                                 cta_tutor_ids, 
    #                                 activity_bkt,
    #                                 tutorID_to_kc_dict,
    #                                 student_id,
    #                                 skill_to_number_map,
    #                                 uniq_skill_groups,
    #                                 skill_group_to_activity_map,
    #                                 CONSTANTS["ACTION_SIZE"],
    #                                 None)

    # init_p_know                   = env.reset()
    # init_avg_p_know               = np.mean(np.array(init_p_know))
    # target_avg_p_know             = CONSTANTS["TARGET_P_KNOW"]
    # final_p_know                  = init_p_know.copy()
    # final_avg_p_know              = init_avg_p_know
    # CONSTANTS["TARGET_REWARD"]    = 1000 * (target_avg_p_know - init_avg_p_know)

    # clear log files: ppo_logs
    # clear_files("ppo", True)

    # mkdir('.', 'checkpoints')
    
    # writer = SummaryWriter(comment="ppo_" + args.name)

    # # Autodetect CUDA
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu:0")
    # print('Device:', device)

    # # Prepare environments
    # envs = [make_env(i+1, 
    #                 initial_state, 
    #                 cta_tutor_ids, 
    #                 activity_bkt, 
    #                 tutorID_to_kc_dict, 
    #                 student_id, 
    #                 skill_to_number_map, 
    #                 uniq_skill_groups, 
    #                 skill_group_to_activity_map, 
    #                 CONSTANTS["ACTION_SIZE"]) for i in range(CONSTANTS["NUM_ENVS"])]
    # envs = SubprocVecEnv(envs)
    
    # num_inputs  = CONSTANTS["STATE_SIZE"]
    # n_actions = CONSTANTS["ACTION_SIZE"]

    # model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[num_inputs], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=n_actions)
    # # model.load_state_dict(torch.load("checkpoints/RoboTutor_best_+804.453_30720.dat"))
    # frame_idx = 0
    # train_epoch = 0
    # best_reward = None

    # # returns a state list, one for each env we make. dim: NUM_ENVS * STATE_SIZE
    # state       = envs.reset()
    # early_stop  = False

    # while not early_stop:
    #     # lists to store training data
    #     log_probs       = []
    #     critic_values   = []
    #     states          = []
    #     actions         = []
    #     rewards         = []
    #     dones           = []
    #     timesteps       = 0

    #     for _ in range(CONSTANTS["PPO_STEPS"]):
    #         timesteps += 1
    #         state = torch.Tensor(state).to(device)
    #         policy, critic_value = model.forward(state)
    #         # softmax ensures actions add up to one which is a requirement for probabilities
    #         policy = F.softmax(policy, dim=1)
    #         action_probs = torch.distributions.Categorical(policy)
    #         action = action_probs.sample()
    #         next_state, reward, student_response, done, posterior = envs.step(action.cpu().numpy(), [timesteps] * CONSTANTS["NUM_ENVS"], [CONSTANTS["MAX_TIMESTEPS"]] * CONSTANTS["NUM_ENVS"])
    #         log_prob = action_probs.log_prob(action)
    #         log_probs.append(log_prob)
    #         critic_values.append(critic_value)
    #         rewards.append(torch.Tensor(reward).unsqueeze(1).to(device))
    #         dones.append(torch.Tensor(1 - done).unsqueeze(1).to(device))
    #         states.append(state)
    #         actions.append(action)
    #         state = next_state
    #         frame_idx += 1

    #     _, critic_value_ = model(torch.Tensor(next_state).to(device))
    #     returns = compute_gae(critic_value_, rewards, dones, critic_values, CONSTANTS["GAMMA"], CONSTANTS["GAE_LAMBDA"])
    #     returns         = torch.cat(returns).detach()
    #     log_probs       = torch.cat(log_probs).detach()
    #     critic_values   = torch.cat(critic_values).detach()
    #     states          = torch.cat(states)
    #     actions         = torch.cat(actions)
    #     advantage       = returns - critic_values
    #     advantage       = normalize(advantage)

    #     # According to PPO paper: (states, actions, log_probs, returns, advantage) is together referred to as a "trajectory"
    #     ppo_update(model, frame_idx, states, actions, log_probs, returns, advantage, CONSTANTS)
    #     train_epoch += 1
        
    #     if train_epoch % CONSTANTS["TEST_EPOCHS"] == 0:
    #         env = StudentEnv(initial_state, cta_tutor_ids, activity_bkt, tutorID_to_kc_dict, student_id, skill_to_number_map, uniq_skill_groups, skill_group_to_activity_map, CONSTANTS["ACTION_SIZE"], None)
    #         test_reward = np.mean([test_env(env, model, device, CONSTANTS) for _ in range(CONSTANTS["NUM_TESTS"])])
    #         writer.add_scalar("test_rewards", test_reward, frame_idx)
    #         print('Frame %s. reward: %s' % (frame_idx, test_reward))
    #         print("FINAL AVG P(Know) after run ", CONSTANTS["RUN_NUM"], ": ", (test_reward/1000 + init_avg_p_know))
    #         print("FINAL P(Know): after run ", CONSTANTS["RUN_NUM"], ": ", env.state)
    #         final_p_know = env.state
    #         final_avg_p_know = np.mean(np.array(final_p_know))
    #         # Save a checkpoint every time we achieve a best reward
    #         if best_reward is None or best_reward < test_reward:
    #             if best_reward is not None:
    #                 print("Best reward updated: %.3f -> %.3f Target reward: %.3f" % (best_reward, test_reward, CONSTANTS["TARGET_REWARD"]))
    #                 name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
    #                 fname = os.path.join('.', 'checkpoints', name)
    #                 torch.save(model.state_dict(), fname)
    #             best_reward = test_reward
    #         if test_reward > CONSTANTS["TARGET_REWARD"]: 
    #             early_stop = True

    # print("INIT P(Know): \n", init_p_know)
    # print("FINAL P(Know): \n", final_p_know)
    # print("IMPROVEMENT PER SKILL: \n", np.array(final_p_know) - np.array(init_p_know))
    # print("INIT AVG P(KNOW): ", init_avg_p_know)
    # print("FINAL AVG P(KNOW): ", final_avg_p_know)
    # print("TOTAL RUNS: ", CONSTANTS["RUN_NUM"])