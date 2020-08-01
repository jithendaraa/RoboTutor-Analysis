# Imports
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

# empty contents of all txt files in "ppo_logs" (which has been gitignored due to large file size) folder, set to False if you don't want to empty contents
clear_files("ppo", True)

CONSTANTS = {
                "STUDENT_ID"            : "new_student",
                "ENV_ID"                : "RoboTutor",
                "COMPARE_STUDENT_IDS"   : ['VPRQEF_101'],
                "STATE_SIZE"            : 18,
                "ACTION_SIZE"           : 33,
                "FC1_DIMS"              : 256,
                "LEARNING_RATE"         : 1e-4,
                "MAX_TIMESTEPS"         : 150,
                "PLOT_TIMESTEPS"        : 300
            }

if __name__ == '__main__':

    compare_student_ids = []
    kc_list, num_skills, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
    # activity_bkt, activity_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(1.0, train_students=[CONSTANTS["STUDENT_ID"]] + CONSTANTS["COMPARE_STUDENT_IDS"])
    activity_bkt, tutorID_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(0.5, ['VPRQEF_101'])
    activity_bkt.set_learning_progress(CONSTANTS["STUDENT_ID"], activity_bkt.activity_learning_progress['VPRQEF'], activity_bkt.know[0])

    student_id          = student_id_to_number_map[CONSTANTS["STUDENT_ID"]]
    initial_state       = np.array(activity_bkt.know[student_id])

    for compare_student_id in CONSTANTS["COMPARE_STUDENT_IDS"]:
        compare_student_ids.append(student_id_to_number_map[compare_student_id])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=CONSTANTS["ENV_ID"], help="Environment name to use, default=" + CONSTANTS["ENV_ID"])
    parser.add_argument("-d", "--deterministic", default=True, action="store_true", help="enable deterministic actions")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # Init environment
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

    # Load model
    num_inputs  = [CONSTANTS["STATE_SIZE"]]
    num_outputs = CONSTANTS["ACTION_SIZE"]
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[num_inputs], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=num_outputs)
    model.load_state_dict(torch.load("checkpoints/" + args.model))
    print("Loaded model at checkpoints/" + str(args.model))

    """ 
        Use loaded model so we can play according to learned policy
        Returns 
            total_reward (float): 
                Total rewards accumulated till the end of episode, 

            posterior (Torch Tensor in the GPU):
                P(Know) at the end of episode after student learns from the RL policy

            learning_progress (dict) 
                learning_progress["VPRQEF_101"] will be a 2D list which and learning_progress["VPRQEF_101"][i] will contain P(Know) for each skill after opportunity i
    """
    total_reward, posterior, learning_progress = play_env(env, model, device, CONSTANTS, deterministic=False)

    new_student_avg = []
    student_avgs = []

    # Push the avg P(Know) of CONSTANTS["STUDENT_ID"] into new_student_avg after each opportunity 
    for know in learning_progress[CONSTANTS["STUDENT_ID"]]:
        avg_know = np.mean(np.array(know))
        new_student_avg.append(avg_know)
    
    # Push avg P(Know) of each student in CONSTANTS["COMPARE_STUDENT_IDS"] into student_avgs after each opportunity. student_avgs is thus a 2d list
    for compare_student_id in CONSTANTS["COMPARE_STUDENT_IDS"]:
        student_avg = []
        for know in learning_progress[compare_student_id]:
            avg_know = np.mean(np.array(know))
            student_avg.append(avg_know)
        student_avgs.append(student_avg)

    # Plot "new_student" performance (which uses RL policy) versus performance of students in CONSTANTS["COMPARE_STUDENT_IDS"] which uses RoboTutor's Policy
    plot_learning(learning_progress, CONSTANTS["COMPARE_STUDENT_IDS"], CONSTANTS["PLOT_TIMESTEPS"], new_student_avg, student_avgs, algo="PPO")

    # Print final Avg P(Know) after playing the policy learned by the PPO algorithm
    posterior = posterior.cpu().numpy()
    avg_p_know = np.mean(posterior)
    print("Avg. P(Know): ", avg_p_know)

    