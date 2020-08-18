# Imports
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

from actor_critic_agent import ActorCritic
from environment import StudentEnv
from student_simulator import StudentSimulator

from helper import *
from reader import *
from ppo_helper import play_env

CONSTANTS = {
                "STUDENT_ID"            : 'new_student',
                "ENV_ID"                : "RoboTutor",
                "COMPARE_STUDENT_IDS"   : ['5A27001753'],
                "STATE_SIZE"            : 22,
                "ACTION_SIZE"           : 43,
                "FC1_DIMS"              : 256,
                "LEARNING_RATE"         : 1e-4,
                "MAX_TIMESTEPS"         : 10,
                "PLOT_TIMESTEPS"        : 300,
                'STUDENT_MODEL_NAME'    : 'hotDINA_full',
                'VILLAGE'               : '130',
                'NUM_OBS'               : '100',
                'PATH_TO_ACTIIVTY_TABLE': 'Data/Activity_table_v4.1_22Apr2020.pkl',
                'DETERMINISTIC'         : True
            }

def set_constants(args):
    CONSTANTS['NUM_OBS']            = args.observations
    CONSTANTS['VILLAGE']            = args.village
    CONSTANTS['STUDENT_ID']         = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model file to load")
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village", default=CONSTANTS['VILLAGE'], help="Village number to play learnt policy on")
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument("-smn", "--student_model_name", help="Student model that the Student Simulator uses", default=CONSTANTS['STUDENT_MODEL_NAME'])
    parser.add_argument("-e", "--env", default=CONSTANTS["ENV_ID"], help="Environment name to use, default=" + CONSTANTS["ENV_ID"])
    parser.add_argument("-d", "--deterministic", default=CONSTANTS['DETERMINISTIC'], action="store_true", help="enable deterministic actions")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir", default='Disabled')
    args = parser.parse_args()
    set_constants(args)

    path = '../'
    kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data(path)

    student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"],
                                        path=path)

    uniq_student_ids = student_simulator.uniq_student_ids
    
    if CONSTANTS['STUDENT_ID'] not in uniq_student_ids:
        CONSTANTS['STUDENT_ID'] = uniq_student_ids[0]
    
    compare_student_nums = []
    for student_id in CONSTANTS['COMPARE_STUDENT_IDS']:
        compare_student_nums.append(uniq_student_ids.index(student_id))

    student_num = uniq_student_ids.index(CONSTANTS["STUDENT_ID"])
    compare_student_nums = []
    for compare_student_id in CONSTANTS["COMPARE_STUDENT_IDS"]:
        compare_student_nums.append(student_simulator.uniq_student_ids.index(compare_student_id))

    data_dict = get_data_dict(uniq_student_ids, student_simulator.kc_list, path)
    student_simulator.update_on_log_data(data_dict, train_students=compare_student_nums, plot=False, bayesian_update=True)
    
    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # Init environment
    env = StudentEnv(student_simulator=student_simulator,
                    skill_groups=uniq_skill_groups,
                    skill_group_to_activity_map = skill_group_to_activity_map,
                    action_size=CONSTANTS["ACTION_SIZE"],
                    student_id=CONSTANTS["STUDENT_ID"])
    env.checkpoint()
    # empty contents of all txt files in "ppo_logs" (which has been gitignored due to large file size) folder, set to False if you don't want to empty contents
    clear_files("ppo", True, path='RL_agents/')

    # # Load model
    num_inputs  = CONSTANTS["STATE_SIZE"]
    n_actions = CONSTANTS["ACTION_SIZE"]
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[num_inputs], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=n_actions)
    model.load_state_dict(torch.load("checkpoints/" + args.model))
    print("Loaded model at checkpoints/" + str(args.model))

    total_reward, posterior, learning_progress = play_env(env, model, device, CONSTANTS, deterministic=CONSTANTS['DETERMINISTIC'])
    
    # Push the avg P(Know) of CONSTANTS["STUDENT_ID"] into new_student_avg after each opportunity 
    new_student_avg = []
    for know in learning_progress[student_num]:
        avg_know = np.mean(np.array(know))
        new_student_avg.append(avg_know)
    
    # Push avg P(Know) of each student in CONSTANTS["COMPARE_STUDENT_IDS"] into student_avgs after each opportunity. student_avgs is thus a 2d list
    student_avgs = []
    for compare_student_num in compare_student_nums:
        student_avg = []
        for know in learning_progress[compare_student_num]:
            avg_know = np.mean(np.array(know))
            student_avg.append(avg_know)
        student_avgs.append(student_avg)

    # Plot "new_student" performance (which uses RL policy) versus performance of students in CONSTANTS["COMPARE_STUDENT_IDS"] which uses RoboTutor's Policy
    plot_learning(learning_progress, compare_student_nums, CONSTANTS["PLOT_TIMESTEPS"], new_student_avg, algo="PPO")

    # Print final Avg P(Know) after playing the policy learned by the PPO algorithm
    posterior = posterior.cpu().numpy()
    avg_p_know = np.mean(posterior)
    print("All unique students under this village: ", uniq_student_ids)
    print("Avg. P(Know): ", avg_p_know)