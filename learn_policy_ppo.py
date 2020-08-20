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
from student_simulator import StudentSimulator
from tutor_simulator import TutorSimulator
from environment import StudentEnv

from RL_agents.ppo_helper import *

from reader import *
from helper import *

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
    'PPO_EPOCHS'                :   10,
    'TEST_EPOCHS'               :   10,
    'NUM_TESTS'                 :   10,
    'GAE_LAMBDA'                :   0.95,
    "MINI_BATCH_SIZE"           :   32,
    "MAX_TIMESTEPS"             :   50,
    'GAMMA'                     :   0.99,
    'GAE_LAMBDA'                :   0.95,
    'PPO_EPSILON'               :   0.2,
    "CRITIC_DISCOUNT"           :   0.5, 
    "ENTROPY_BETA"              :   0.001,
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
    CONSTANTS['AREA_ROTATION'] = args.area_rotation

    if args.type == 1:
        CONSTANTS['STATE_SIZE'] = 22
        CONSTANTS['ACTION_SIZE'] = 3

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument('-smn', '--student_model_name', help="Student model name")
    parser.add_argument('--ppo_steps', help="PPO Steps", default=CONSTANTS['PPO_STEPS'])
    parser.add_argument('-ar', '--area_rotation', help="Area rotation sequence like L-N-L-S", default=CONSTANTS['AREA_ROTATION'])
    parser.add_argument('-arc', '--area_rotation_constraint', help="Should questions be constrained like lit-num-lit-stories? True/False", default=True)
    parser.add_argument('-tc', '--transition_constraint', help="Should transition be constrained to prev,same,next, next-next? True/False", default=True)
    parser.add_argument('-m', '--model', help="Model file to load from checkpoints directory, if any")
    parser.add_argument('-nsp', '--new_student_params', help="The model params new_student has to start with; enter student_id")
    
    args = parser.parse_args()
    set_constants(args)
    return args



if __name__ == '__main__':
    
    args = arg_parser()
    student_id = CONSTANTS['STUDENT_ID']
    state_size  = CONSTANTS["STATE_SIZE"]
    action_size = CONSTANTS["ACTION_SIZE"]
    
    student_simulator = StudentSimulator(village=args.village_num, 
                                        observations=args.observations, 
                                        student_model_name=args.model, 
                                        new_student_params=args.new_student_params)
    
    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(student_id)

    env = StudentEnv(student_simulator, action_size, student_id, 1, args.type)
    env.checkpoint()
    
    init_p_know = env.reset()
    init_avg_p_know = np.mean(np.array(init_p_know))
    target_avg_p_know = CONSTANTS["TARGET_P_KNOW"]
    final_p_know = init_p_know.copy()
    final_avg_p_know = init_avg_p_know
    CONSTANTS["TARGET_REWARD"] = 1000 * (target_avg_p_know - init_avg_p_know)

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    print('Device:', device)
    
    # Prepare environments
    envs = [make_env(i+1, student_simulator, student_id, action_size,type=args.type) for i in range(CONSTANTS["NUM_ENVS"])]
    envs = SubprocVecEnv(envs)
    envs.checkpoint()

    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=action_size, type=args.type)
    
    tutor_simulator = TutorSimulator(LOW_PERFORMANCE_THRESHOLD, 
                                    MID_PERFORMANCE_THRESHOLD, 
                                    HIGH_PERFORMANCE_THRESHOLD, 
                                    area_rotation=CONSTANTS['AREA_ROTATION'], 
                                    type=args.type)
    
    if args.model != None:
        model.load_state_dict(torch.load("checkpoints/"+args.model))
    
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
            print(policy)

            break
        break
    
