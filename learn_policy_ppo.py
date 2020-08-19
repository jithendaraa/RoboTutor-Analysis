import os
import argparse
import numpy as np
from pathlib import Path
import pickle
import math

import torch 
import torch.nn.functional as F

from lib.common import mkdir
from lib.multiprocessing_env import SubprocVecEnv

from RL_agents.actor_critic_agent import ActorCritic
from environment import StudentEnv
from student_simulator import StudentSimulator

from RL_agents.ppo_helper import *

from reader import *

CONSTANTS = {
    'NUM_OBS'                   :   '100',
    'VILLAGE'                   :   '130',
    'STUDENT_ID'                :   'new_student',
    'STUDENT_MODEL_NAME'        :   'hotDINA_skill',
    'AREA_ROTATION'             :   'L-N-L-S',
    'START_POS'                 :   '0,0',
    'NUM_ENVS'                  :   2,
    'AGENT_TYPE'                :   1,
    'AREA_ROTATION_CONSTRAINT'  :   True,
    'TRANSITION_CONSTRAINT'     :   True,
    "TARGET_P_KNOW"             :   0.5,
    "LEARNING_RATE"             :   1e-4,
    "FC1_DIMS"                  :   256,
    'PPO_STEPS'                 :   10,
    'PPO_EPOCHS'                :   5,
    'GAE_LAMBDA'                :   0.95,
    "MINI_BATCH_SIZE"           :   32,
    "MAX_TIMESTEPS"             :   50,
}

# Current RoboTutor thresholds
LOW_PERFORMANCE_THRESHOLD = 0.5
MID_PERFORMANCE_THRESHOLD = 0.83
HIGH_PERFORMANCE_THRESHOLD = 0.9
LOW_LENIENT_PERFORMANCE_THRESHOLD = 0.4
MID_LENIENT_PERFORMANCE_THRESHOLD = 0.55
HIGH_LENIENT_PERFORMANCE_THRESHOLD = 0.7

def set_constants(args):
    CONSTANTS['NUM_OBS'] = args.observations
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['AGENT_TYPE'] = args.type
    CONSTANTS['STUDENT_ID'] = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['PPO_STEPS'] = args.ppo_steps
    CONSTANTS['AREA_ROTATION_CONSTRAINT'] = args.area_rotation_constraint
    CONSTANTS['TRANSITION_CONSTRAINT'] = args.transition_constraint

    if args.type == 1:
        CONSTANTS['STATE_SIZE'] = 22
        CONSTANTS['ACTION_SIZE'] = 3
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument('-smn', '--student_model_name', help="Student model name")
    parser.add_argument('--ppo_steps', help="PPO Steps", default=CONSTANTS['PPO_STEPS'])
    parser.add_argument('-arc', '--area_rotation_constraint', help="Should questions be constrained like lit-num-lit-stories? True/False", default=CONSTANTS['AREA_ROTATION_CONSTRAINT'])
    parser.add_argument('-tc', '--transition_constraint', help="Should transition be constrained to prev,same,next, next-next? True/False", default=CONSTANTS['TRANSITION_CONSTRAINT'])
    args = parser.parse_args()
    set_constants(args)

    literacy_matrix, math_matrix, stories_matrix, literacy_counts, math_counts, stories_counts = read_activity_matrix()
    
    kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
    student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"])
    env = StudentEnv(student_simulator=student_simulator,
                    skill_groups=uniq_skill_groups,
                    skill_group_to_activity_map = skill_group_to_activity_map,
                    action_size=CONSTANTS["ACTION_SIZE"],
                    student_id=CONSTANTS["STUDENT_ID"],
                    type=args.type)
    env.checkpoint()

    t1, t2, t3 = LOW_PERFORMANCE_THRESHOLD ,MID_PERFORMANCE_THRESHOLD, HIGH_PERFORMANCE_THRESHOLD

    xpos = int(CONSTANTS['START_POS'].split(',')[0])
    ypos = int(CONSTANTS['START_POS'].split(',')[1])
    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(CONSTANTS['STUDENT_ID'])
    literacy_activity = literacy_matrix[xpos][ypos]
    
    init_p_know = env.reset()
    init_avg_p_know = np.mean(np.array(init_p_know))
    target_avg_p_know = CONSTANTS["TARGET_P_KNOW"]
    final_p_know = init_p_know.copy()
    final_avg_p_know = init_avg_p_know
    CONSTANTS["TARGET_REWARD"] = 1000 * (target_avg_p_know - init_avg_p_know)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    print('Device:', device)

    num_inputs  = CONSTANTS["STATE_SIZE"]
    n_actions = CONSTANTS["ACTION_SIZE"]
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[num_inputs], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=n_actions, type=args.type)
    
    # Prepare environments
    envs = [make_env(i+1, 
                    student_simulator, 
                    CONSTANTS["STUDENT_ID"], 
                    uniq_skill_groups, 
                    skill_group_to_activity_map, 
                    CONSTANTS["ACTION_SIZE"]) for i in range(CONSTANTS["NUM_ENVS"])]
    envs = SubprocVecEnv(envs)
    envs.checkpoint()