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
    'NUM_OBS'       :   '50',
    'VILLAGE'       :   '130',
    'TYPE'          :   2,
    'STUDENT_ID'    :   'new_student',
    'AREA_ROTATION' :   'L-N-L-S',
    # Current RoboTutor Thresholds
    'LOW_PERFORMANCE_THRESHOLD'         :   0.5,
    'MID_PERFORMANCE_THRESHOLD'         :   0.83,
    'HIGH_PERFORMANCE_THRESHOLD'        :   0.9,
    'LOW_LENIENT_PERFORMANCE_THRESHOLD' :   0.4,
    'MID_LENIENT_PERFORMANCE_THRESHOLD' :   0.55,
    'HIGH_LENIENT_PERFORMANCE_THRESHOLD':   0.7,
}

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument("-t", "--type", default=CONSTANTS["TYPE"], help="Village to train on (not applicable for Activity BKT)")
    args = parser.parse_args()
    return args

def set_constants(args):
    CONSTANTS['NUM_OBS'] = args.observations
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['AGENT_TYPE'] = args.type

    if args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = 22 + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256

def local_policy_evaluation(attempt_num):
    local_policy_score = 0.0 

    # do stuff

    return local_policy_score

def global_policy_evaluation():
    global_policy_score = 0.0
    timesteps = int(CONSTANTS['NUM_OBS'])
    for i in range(timesteps):
        global_policy_score += local_policy_evaluation(i)

    return global_policy_score


if '__name__' == '__main__':
    
    args = argparser()
    set_constants(args)

    student_id = CONSTANTS['STUDENT_ID']
    area_rotation = CONSTANTS['AREA_ROTATION']

    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=student_id, prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])

    pass