# Imports
import sys
import os
sys.path.append(os.getcwd()+'/lib')
import argparse
import math
import random
import numpy as np
from pathlib import Path
import pickle

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
    'NUM_SKILLS'                        :   None,
    'STATE_SIZE'                        :   None,
    'ACTION_SIZE'                       :   None,
    'ENV_ID'                            :   "RoboTutor",    
    'NUM_OBS'                           :   '100',
    'VILLAGE'                           :   '130',
    'STUDENT_ID'                        :   'new_student',
    'STUDENT_MODEL_NAME'                :   'hotDINA_skill',
    'AREA_ROTATION'                     :   'L-N-L-S',
    'START_POS'                         :   '0,0',
    'AVG_OVER_RUNS'                     :   20,
    'AGENT_TYPE'                        :   None,
    'AREA_ROTATION_CONSTRAINT'          :   True,
    'TRANSITION_CONSTRAINT'             :   True,
    "LEARNING_RATE"                     :   1e-4,
    "FC1_DIMS"                          :   1024,
    "FC2_DIMS"                          :   2048,
    'FC3_DIMS'                          :   1024,
    "MAX_TIMESTEPS"                     :   50,
    'PRINT_STUDENT_PARAMS'              :   True,
    'CLEAR_LOGS'                        :   True,
    'DETERMINISTIC'                     :   False,
    'USES_THRESHOLDS'                   :   False,    
    # Current RoboTutor Thresholds
    'LOW_PERFORMANCE_THRESHOLD'         :   0.5,
    'MID_PERFORMANCE_THRESHOLD'         :   0.83,
    'HIGH_PERFORMANCE_THRESHOLD'        :   0.9,
    'LOW_LENIENT_PERFORMANCE_THRESHOLD' :   0.4,
    'MID_LENIENT_PERFORMANCE_THRESHOLD' :   0.55,
    'HIGH_LENIENT_PERFORMANCE_THRESHOLD':   0.7,
}


def set_constants(args):
    CONSTANTS['NUM_OBS'] = args.observations
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['STUDENT_ID'] = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['DETERMINISTIC'] = args.deterministic
    CONSTANTS['NEW_STUDENT_PARAMS'] = args.new_student_params

    if args.type == None:
        CONSTANTS['STATE_SIZE'] = 22
        CONSTANTS['USES_THRESHOLDS'] = None
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = False

    if args.type == 1:  # State size: number of KC's; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = 22
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True
    
    elif args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = 22 + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True

    elif args.type == 3:
        CONSTANTS['STATE_SIZE'] = 22 + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 4    # prev, same, next, next_next
        CONSTANTS['USES_THRESHOLDS'] = False
    
    elif args.type == 4:
        CONSTANTS['STATE_SIZE'] = 22 + 1 
        CONSTANTS['ACTION_SIZE'] = None
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = True
    
    elif args.type == 5:
        CONSTANTS['STATE_SIZE'] = 22 + 1 
        CONSTANTS['ACTION_SIZE'] = None
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = False


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS['VILLAGE'], help="Village number to play learnt policy on")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument('-mt', '--max_timesteps', help="Total questions that will be given to the student/RL agent", type=int, default=CONSTANTS['MAX_TIMESTEPS'])
    parser.add_argument("-m", "--model", help="Model file to load")
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument("-smn", "--student_model_name", help="Student model that the Student Simulator uses", default=CONSTANTS['STUDENT_MODEL_NAME'])
    parser.add_argument("-d", "--deterministic", default=CONSTANTS['DETERMINISTIC'], action="store_true", help="enable deterministic actions")
    parser.add_argument('-nsp', '--new_student_params', help="The model params new_student has to start with; enter student_id")
    args = parser.parse_args()
    set_constants(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    print('Device:', device)        # Autodetect CUDA
    return args


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    args = arg_parser()

    student_id = CONSTANTS['STUDENT_ID']
    area_rotation = CONSTANTS['AREA_ROTATION']
    # village 130: ['5A27001753', '5A27001932', '5A28002555', '5A29000477', '6105000515', '6112001212', '6115000404', '6116002085', 'new_student']

    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=args.new_student_params, prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])



    