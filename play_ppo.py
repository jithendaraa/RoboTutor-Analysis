import sys
sys.path.append("lib")

import argparse
import math
import os
import random
import numpy as np

import torch 
import torch.nn.functional as F

from actor_critic_agent import ActorCritic
from environment import StudentEnv
from helper import read_data, clear_files, plot_learning
from student_simulator import train_on_obs
from ppo_helper import play_env

clear_files("ppo", True)

CONSTANTS = {
                "STUDENT_ID"            : "new_student",
                "ACTION_SIZE"           : 33,
                "ENV_ID"                : "RoboTutor",
                "FC1_DIMS"              : 256,
                "LEARNING_RATE"         : 1e-4,
                "MAX_TIMESTEPS"         : 150,
                "COMPARE_STUDENT_IDS"   : ['VPRQEF_101', 'CQCKBY_105', 'QNPZWV_106', 'PTQUQC_175', 'ZKTUNM_171', 'HNAWRP_169'],
                "PLOT_TIMESTEPS"        : 300
            }

if __name__ == '__main__':

    compare_student_ids = []
    kc_list, num_skills, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
    activity_bkt, activity_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(1.0, train_students=[CONSTANTS["STUDENT_ID"]] + CONSTANTS["COMPARE_STUDENT_IDS"])

    student_id          = student_id_to_number_map[CONSTANTS["STUDENT_ID"]]
    for compare_student_id in CONSTANTS["COMPARE_STUDENT_IDS"]:
        compare_student_ids.append(student_id_to_number_map[compare_student_id])
    
    initial_state       = np.array(activity_bkt.know[student_id])

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=CONSTANTS["ENV_ID"], help="Environment name to use, default=" + CONSTANTS["ENV_ID"])
    parser.add_argument("-d", "--deterministic", default=True, action="store_true", help="enable deterministic actions")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    env = StudentEnv(initial_state, 
                        cta_tutor_ids, 
                        activity_bkt,
                        tutorID_to_kc_dict,
                        student_id,
                        skill_to_number_map,
                        uniq_skill_groups,
                        skill_group_to_activity_map,
                        CONSTANTS["ACTION_SIZE"],
                        None)

    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[num_inputs], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=CONSTANTS["ACTION_SIZE"])
    model.load_state_dict(torch.load("checkpoints/" + args.model))
    print("Loaded model at checkpoints/" + str(args.model))

    total_reward, posterior, learning_progress = play_env(env, model, device, CONSTANTS, deterministic=False)

    new_student_avg = []
    student_avgs = []

    for know in learning_progress[CONSTANTS["STUDENT_ID"]]:
        avg_know = np.mean(np.array(know))
        new_student_avg.append(avg_know)
    
    for compare_student_id in CONSTANTS["COMPARE_STUDENT_IDS"]:
        student_avg = []
        for know in learning_progress[compare_student_id]:
            avg_know = np.mean(np.array(know))
            student_avg.append(avg_know)
        student_avgs.append(student_avg)

    plot_learning(learning_progress, CONSTANTS["COMPARE_STUDENT_IDS"], CONSTANTS["PLOT_TIMESTEPS"], new_student_avg, student_avgs, algo="PPO")

    posterior = posterior.cpu().numpy()
    avg_p_know = np.mean(posterior)
    print("Avg. P(Know): ", avg_p_know)

    