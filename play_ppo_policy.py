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
    "MAX_TIMESTEPS"                     :   120,
    'PRINT_STUDENT_PARAMS'              :   True,
    'CLEAR_LOGS'                        :   True,
    'DETERMINISTIC'                     :   False,
    'USES_THRESHOLDS'                   :   False,  
    'AVG_OVER_RUNS'                     :   10,  
    'STUDENT_SPEC_MODEL'                :   False,
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
    CONSTANTS['MAX_TIMESTEPS'] = args.max_timesteps
    CONSTANTS['STUDENT_ID'] = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['DETERMINISTIC'] = args.deterministic
    CONSTANTS['NEW_STUDENT_PARAMS'] = args.new_student_params
    CONSTANTS['STUDENT_SPEC_MODEL'] = args.student_spec_model

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
    parser.add_argument('--student_spec_model', help="Set true if you want to use model specific to the student", default=CONSTANTS['STUDENT_SPEC_MODEL'])
    parser.add_argument('-mt', '--max_timesteps', help="Total questions that will be given to the student/RL agent", type=int, default=CONSTANTS['MAX_TIMESTEPS'])
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument("-smn", "--student_model_name", help="Student model that the Student Simulator uses", default=CONSTANTS['STUDENT_MODEL_NAME'])
    parser.add_argument("-d", "--deterministic", default=CONSTANTS['DETERMINISTIC'], help="enable deterministic actions")
    parser.add_argument('-nsp', '--new_student_params', help="The model params new_student has to start with; enter student_id")
    parser.add_argument('-ar', '--area_rotation', help="Area rotation sequence like L-N-L-S", default=CONSTANTS['AREA_ROTATION'])
    parser.add_argument('-arc', '--area_rotation_constraint', help="Should questions be constrained like lit-num-lit-stories? True/False", default=True)
    parser.add_argument('-tc', '--transition_constraint', help="Should transition be constrained to prev,same,next, next-next? True/False", default=True)
    args = parser.parse_args()
    set_constants(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    print('Device:', device)        # Autodetect CUDA
    return args

def set_action_constants(type, tutor_simulator):
    if args.type == 4 or args.type == 5 or args.type == None:
        num_literacy_acts = len(tutor_simulator.literacy_activities)
        num_math_acts = len(tutor_simulator.math_activities)
        num_story_acts = len(tutor_simulator.story_activities)
        CONSTANTS['ACTION_SIZE'] = [num_literacy_acts, num_math_acts, num_story_acts]
        CONSTANTS['NUM_LITERACY_ACTS'], CONSTANTS['NUM_MATH_ACTS'], CONSTANTS['NUM_STORY_ACTS'] = CONSTANTS['ACTION_SIZE'][0], CONSTANTS['ACTION_SIZE'][1], CONSTANTS['ACTION_SIZE'][2]
        CONSTANTS['LITERACY_ACTS'], CONSTANTS['MATH_ACTS'], CONSTANTS['STORY_ACTS'] = tutor_simulator.literacy_activities, tutor_simulator.math_activities, tutor_simulator.story_activities
        if args.type == 5 or args.type == None:  
            CONSTANTS['ACTION_SIZE'] = num_literacy_acts + num_math_acts + num_story_acts
    else:
        pass

def evaluate_current_RT_thresholds(plots=True, prints=True, avg_over_runs=CONSTANTS['AVG_OVER_RUNS']):

    LOW_PERFORMANCE_THRESHOLD = CONSTANTS['LOW_PERFORMANCE_THRESHOLD']
    MID_PERFORMANCE_THRESHOLD = CONSTANTS['MID_PERFORMANCE_THRESHOLD']
    HIGH_PERFORMANCE_THRESHOLD = CONSTANTS['HIGH_PERFORMANCE_THRESHOLD']

    LOW_LENIENT_PERFORMANCE_THRESHOLD = CONSTANTS['LOW_LENIENT_PERFORMANCE_THRESHOLD']
    MID_LENIENT_PERFORMANCE_THRESHOLD = CONSTANTS['MID_LENIENT_PERFORMANCE_THRESHOLD']
    HIGH_LENIENT_PERFORMANCE_THRESHOLD = CONSTANTS['HIGH_LENIENT_PERFORMANCE_THRESHOLD']
    
    avg_performance_ys = []
    avg_lenient_performance_ys = []
    label1 = "Current RT Thresholds(" + str(LOW_PERFORMANCE_THRESHOLD) + ", " + str(MID_PERFORMANCE_THRESHOLD) + ", " + str(HIGH_PERFORMANCE_THRESHOLD) + ")"
    label2 = "Current Lenient RT Thresholds(" + str(LOW_LENIENT_PERFORMANCE_THRESHOLD) + ", " + str(MID_LENIENT_PERFORMANCE_THRESHOLD) + ", " + str(HIGH_LENIENT_PERFORMANCE_THRESHOLD) + ")"
    for _ in range(avg_over_runs):
        student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=CONSTANTS['NEW_STUDENT_PARAMS'], prints=False)
        tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], CONSTANTS['AREA_ROTATION'], type=1, thresholds=True)
        performance_ys = evaluate_performance_thresholds(student_simulator, tutor_simulator, prints=prints, CONSTANTS=CONSTANTS)
        
        student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=CONSTANTS['NEW_STUDENT_PARAMS'], prints=False)
        tutor_simulator = TutorSimulator(CONSTANTS['LOW_LENIENT_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_LENIENT_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_LENIENT_PERFORMANCE_THRESHOLD'], CONSTANTS['AREA_ROTATION'], type=1, thresholds=True)
        lenient_performance_ys = evaluate_performance_thresholds(student_simulator, tutor_simulator, prints=prints, CONSTANTS=CONSTANTS)
        
        avg_performance_ys.append(performance_ys)
        avg_lenient_performance_ys.append(lenient_performance_ys)
    
    avg_performance_ys = np.mean(avg_performance_ys, axis=0)
    avg_lenient_performance_ys = np.mean(avg_lenient_performance_ys, axis=0)
    xs = np.arange(len(avg_performance_ys)).tolist()
    if plots:
        file_name = 'plots/Current_RT_Thresholds/village_' + CONSTANTS['VILLAGE'] + '/Current_RT_Thresholds_' + str(CONSTANTS['MAX_TIMESTEPS']) + 'obs_' + CONSTANTS['STUDENT_MODEL_NAME'] + '_' + CONSTANTS['STUDENT_ID'] + '.png'
        plt.title("Current RT policy after " + str(CONSTANTS['MAX_TIMESTEPS']) + " attempts using normal thresholds for " + CONSTANTS['STUDENT_MODEL_NAME'] + '(' + CONSTANTS['STUDENT_ID'] + ')')
        plt.plot(xs, performance_ys, color='r', label=label1)
        plt.plot(xs, lenient_performance_ys, color='b', label=label2)
        plt.xlabel('#Opportunities')
        plt.ylabel('Avg P(Know) across skills')
        plt.legend()
    return xs, performance_ys, lenient_performance_ys


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    args = arg_parser()

    student_id = CONSTANTS['STUDENT_ID']
    area_rotation = CONSTANTS['AREA_ROTATION']
    # village 130: ['5A27001753', '5A27001932', '5A28002555', '5A29000477', '6105000515', '6112001212', '6115000404', '6116002085', 'new_student']

    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=student_id, prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])
    set_action_constants(args.type, tutor_simulator)
    
    state_size  = CONSTANTS["STATE_SIZE"]
    action_size = CONSTANTS["ACTION_SIZE"]

    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(student_id)

    checkpoint_file_name_start = student_id + '~' + args.student_model_name + '~obs_' + args.observations + '~max_timesteps_' + str(args.max_timesteps) + '~village_' + args.village_num + '~type_' + str(args.type) + '~'

    checkpoint_files = os.listdir('./checkpoints')
    checkpoint_file_name = ""

    for file in checkpoint_files:
        if file[:len(checkpoint_file_name_start)] == checkpoint_file_name_start:
            checkpoint_file_name = file
            break

    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=action_size, type=args.type)
    model.load_state_dict(torch.load("checkpoints/" + checkpoint_file_name))
    print("Loaded model at checkpoints/" + checkpoint_file_name)
    
    fig = plt.figure(figsize=(15, 11))

    ax = []
    for i in range(len(student_simulator.uniq_student_ids)):
        ax.append(fig.add_subplot(3,3,i+1))
        ax[i].clear()
    
    for i in range(len(student_simulator.uniq_student_ids)):
        student_num = i
        student_id = student_simulator.uniq_student_ids[i]
        CONSTANTS['NEW_STUDENT_PARAMS'] = student_id
        if student_id == 'new_student': 
            CONSTANTS['NEW_STUDENT_PARAMS'] = None

        if CONSTANTS['STUDENT_SPEC_MODEL']:
            checkpoint_file_name_start = student_id + '~' + args.student_model_name + '~obs_' + args.observations + '~max_timesteps_' + str(args.max_timesteps) + '~village_' + args.village_num + '~type_' + str(args.type) + '~'

            checkpoint_files = os.listdir('./checkpoints')
            checkpoint_file_name = ""
            for file in checkpoint_files:
                if file[:len(checkpoint_file_name_start)] == checkpoint_file_name_start:
                    checkpoint_file_name = file
                    break
           
            if checkpoint_file_name == "":
                checkpoint_file_name_start = 'new_student~' + args.student_model_name + '~obs_' + args.observations + '~max_timesteps_' + str(args.max_timesteps) + '~village_' + args.village_num + '~type_' + str(args.type) + '~'
                for file in checkpoint_files:
                    if file[:len(checkpoint_file_name_start)] == checkpoint_file_name_start:
                        checkpoint_file_name = file
                        break

            model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], n_actions=action_size, type=args.type)
            model.load_state_dict(torch.load("checkpoints/" + checkpoint_file_name))

        student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=student_id, prints=False)

        env = StudentEnv(student_simulator, action_size, student_id, 1, args.type, prints=False, area_rotation=args.area_rotation, CONSTANTS=CONSTANTS)
        env.checkpoint()
        total_reward, posterior, learning_progress = play_env(env, model, device, CONSTANTS, deterministic=args.deterministic)
    
        new_student_avg = []
        for know in learning_progress[student_num]:
            avg_know = np.mean(np.array(know))
            new_student_avg.append(avg_know)

        x = np.arange(1, len(new_student_avg) + 1).tolist()
        xs, threshold_ys, lenient_threshold_ys = evaluate_current_RT_thresholds(plots=False, prints=False)

        ax[i].plot(xs, threshold_ys, color='r', label="Current RT Thresholds")
        ax[i].plot(xs, lenient_threshold_ys, color='b', label="Current lenient RT Thresholds")
        
        RL_perf_text = ""

        if i == 0:
            threshold_text = ""
            lenient_threshold_text = ""
            for j in range(len(xs)):
                threshold_text += str(xs[j]) + "," + str(threshold_ys[j]) + '\n'
                lenient_threshold_text += str(xs[j]) + "," + str(lenient_threshold_ys[j]) + '\n'
            with open("plots/Played plots/threshold.txt", "w") as f:
                f.write(threshold_text)
            with open("plots/Played plots/lenient_threshold.txt", "w") as f:
                f.write(lenient_threshold_text)
        
        for j in range(len(x)):
            RL_perf_text += str(x[j]) + "," + str(new_student_avg[j]) + '\n'
        
        with open("plots/Played plots/Type " + str(args.type) + "/" + student_id + "_RL_perf.txt", "w") as f:
            f.write(RL_perf_text)
        
        ax[i].plot(x, new_student_avg, label="RL Agent", color="black")
        ax[i].set_xlabel("# Opportunities")
        ax[i].set_ylabel("Avg P(Know) across skills")
        ax[i].set_title("Student: " + student_id)
        ax[i].grid()
        ax[i].legend()

    plt.tight_layout()
    plt.grid()
    plt.savefig('../RoboTutor-Analysis/plots/Played plots/Type ' + str(args.type) + '/village_' + args.village_num + '~obs_' + args.observations + '~' + args.student_model_name + '.png')
    plt.show()