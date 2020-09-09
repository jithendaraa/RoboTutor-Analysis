import argparse
import pickle
import os
import sys

import torch 
import torch.nn.functional as F

from RL_agents.actor_critic_agent import ActorCritic
from environment import StudentEnv
from student_simulator import StudentSimulator
from tutor_simulator import TutorSimulator

from helper import *
from reader import *
from RL_agents.ppo_helper import play_env

CONSTANTS = {
    'NUM_OBS'           :   '50',
    'VILLAGE'           :   '130',
    'AGENT_TYPE'              :   2,
    'STUDENT_ID'        :   'new_student',
    'AREA_ROTATION'     :   'L-N-L-S',
    'STUDENT_MODEL_NAME':   'hotDINA_skill',
    'USES_THRESHOLDS'   :   True,
    'STATE_SIZE'        :   22,
    'ACTION_SIZE'       :   4,
    'LEARNING_RATE'     :   5e-4,
    "PRINT_PARAMS_FOR_STUDENT":     False,
    # Current RoboTutor Thresholds
    'LOW_PERFORMANCE_THRESHOLD'         :   0.5,
    'MID_PERFORMANCE_THRESHOLD'         :   0.83,
    'HIGH_PERFORMANCE_THRESHOLD'        :   0.9,
    'LOW_LENIENT_PERFORMANCE_THRESHOLD' :   0.4,
    'MID_LENIENT_PERFORMANCE_THRESHOLD' :   0.55,
    'HIGH_LENIENT_PERFORMANCE_THRESHOLD':   0.7,
}

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument("-t", "--type", default=CONSTANTS["AGENT_TYPE"], help="agent type", type=int)
    parser.add_argument("-smn", "--student_model_name", help="Student model that the Student Simulator uses", default=CONSTANTS['STUDENT_MODEL_NAME'])
    args = parser.parse_args()
    return args

def set_constants(args):
    CONSTANTS['NUM_OBS'] = args.observations
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['AGENT_TYPE'] = args.type
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name

    if args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = 22 + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256

def RT_local_policy_evaluation(data, i, uniq_student_ids, student_simulator):
    RT_local_policy_score = 0.0 

    user = data['users'][i]
    student_id = uniq_student_ids[user]
    act_num = data['items'][i]
    act_name = data['activities'][i]
    
    data_dict = {
        'y'     :   data['y'][i:i+1],
        'items' :   [act_num],
        'users' :   [user]
    }
    student_simulator.update_on_log_data(data_dict, plot=False)
    RT_local_policy_score = np.mean(student_simulator.student_model.alpha[user][-1])
    return RT_local_policy_score

def RL_local_policy_evaluation(attempt_num):
    RL_local_policy_score = 0.0 

    # do stuff

    return RL_local_policy_score

def set_student_simulator(student_id):
    student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=student_id, prints=False)
    return student_simulator

def set_env(student_simulator, student_id, agent_type):
    env = StudentEnv(student_simulator, CONSTANTS['ACTION_SIZE'], student_id, 1, agent_type, prints=False, area_rotation=CONSTANTS['AREA_ROTATION'], CONSTANTS=CONSTANTS)
    env.checkpoint()
    return env

def get_loaded_model(student_id, agent_type):

    checkpoint_file_name_start = student_id + '~' + CONSTANTS['STUDENT_MODEL_NAME'] + '~village_' + CONSTANTS['VILLAGE'] + '~type_' + str(agent_type) + '~'
    checkpoint_files = os.listdir('./checkpoints')
    checkpoint_file_name = ""
    for file in checkpoint_files:
        if file[:len(checkpoint_file_name_start)] == checkpoint_file_name_start:
            checkpoint_file_name = file
            break
    
    state_size = CONSTANTS['STATE_SIZE']
    action_size = CONSTANTS['ACTION_SIZE']
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], fc2_dims=CONSTANTS['FC2_DIMS'], n_actions=action_size, type=agent_type)
    model.load_state_dict(torch.load("checkpoints/" + checkpoint_file_name))
    return model

def global_policy_evaluation(data, uniq_student_ids):
    RT_global_policy_score = 0.0
    RL_global_policy_score = 0.0
    timesteps = int(CONSTANTS['NUM_OBS'])

    for i in range(timesteps):
        user = data['users'][i]
        student_id = uniq_student_ids[user]
        act_num = data['items'][i]
        act_name = data['activities'][i]
        if i == 0 or data['users'][i-1] != data['users'][i]:
            RT_student_simulator = set_student_simulator(student_id)
            RL_student_simulator = set_student_simulator(student_id)
            env = set_env(RL_student_simulator, student_id, args.type)
            model = get_loaded_model(student_id, args.type)

        RT_local_policy_score = RT_local_policy_evaluation(data, i, uniq_student_ids, RT_student_simulator)
        print(RT_local_policy_score)
        # RL_global_policy_score += RL_local_policy_evaluation(data, i)

    return RT_global_policy_score, RL_global_policy_score


if __name__ == '__main__':
    
    args = arg_parser()
    set_constants(args)

    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=CONSTANTS['STUDENT_ID'], prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])

    state_size  = CONSTANTS["STATE_SIZE"]
    action_size = CONSTANTS["ACTION_SIZE"]
    student_id = CONSTANTS['STUDENT_ID']
    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(student_id)

    os.chdir('../hotDINA')
    get_data_command = 'python get_data_for_village_n.py -v ' + args.village_num + ' -o ' + str(args.observations)
    os.system(get_data_command)
    os.chdir('pickles/data')
    
    data_pickle_path = os.getcwd() + '\data' + args.village_num + '_' + str(args.observations) + '.pickle'
    os.chdir('../../../RoboTutor-Analysis')

    # Contains extracted transac table info
    with open(data_pickle_path, "rb") as file:
        try:
            data = pickle.load(file)
        except EOFError:
            print("Error reading pickle")
    
    data['activities'] = []
    for item in data['items']:
        data['activities'].append(uniq_activities[item])
    
    global_policy_evaluation(data, uniq_student_ids)
    
    
    pass