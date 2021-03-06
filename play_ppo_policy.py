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
import tqdm

import torch 
import torch.nn.functional as F

from RL_agents.actor_critic_agent import ActorCritic
from environment import StudentEnv
from simulators.student_simulator import StudentSimulator
from simulators.tutor_simulator import TutorSimulator

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
    'AVG_OVER_RUNS'                     :   50,
    'AGENT_TYPE'                        :   None,
    'AREA_ROTATION_CONSTRAINT'          :   True,
    'TRANSITION_CONSTRAINT'             :   True,
    "LEARNING_RATE"                     :   5e-4,
    "FC1_DIMS"                          :   1024,
    "FC2_DIMS"                          :   2048,
    'FC3_DIMS'                          :   1024,
    "MAX_TIMESTEPS"                     :   100,
    'PRINT_STUDENT_PARAMS'              :   True,
    'CLEAR_LOGS'                        :   True,
    'DETERMINISTIC'                     :   False,
    'USES_THRESHOLDS'                   :   False,  
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
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256

    elif args.type == 3:
        CONSTANTS['STATE_SIZE'] = 22 + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 4    # prev, same, next, next_next
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256
        CONSTANTS['LEARNING_RATE'] = 7e-4
    
    elif args.type == 4:
        CONSTANTS['STATE_SIZE'] = 22 + 1 
        CONSTANTS['ACTION_SIZE'] = None
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = True
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256
        CONSTANTS['LEARNING_RATE'] = 5e-3
    
    elif args.type == 5:
        CONSTANTS['STATE_SIZE'] = 22
        CONSTANTS['ACTION_SIZE'] = None
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = False
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256
        CONSTANTS['LEARNING_RATE'] = 5e-3

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS['VILLAGE'], help="Village number to play learnt policy on")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument('-ssm', '--student_spec_model', help="Set true if you want to use model specific to the student", default=CONSTANTS['STUDENT_SPEC_MODEL'])
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

def evaluate_current_RT_thresholds(plots=True, prints=True, avg_over_runs=100):

    LOW_PERFORMANCE_THRESHOLD = CONSTANTS['LOW_PERFORMANCE_THRESHOLD']
    MID_PERFORMANCE_THRESHOLD = CONSTANTS['MID_PERFORMANCE_THRESHOLD']
    HIGH_PERFORMANCE_THRESHOLD = CONSTANTS['HIGH_PERFORMANCE_THRESHOLD']

    performances_ys = []
    label1 = "Current RT Thresholds(" + str(LOW_PERFORMANCE_THRESHOLD) + ", " + str(MID_PERFORMANCE_THRESHOLD) + ", " + str(HIGH_PERFORMANCE_THRESHOLD) + ")"
    
    for _ in range(avg_over_runs):
        student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=CONSTANTS['NEW_STUDENT_PARAMS'], prints=False)
        tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], CONSTANTS['AREA_ROTATION'], type=1, thresholds=True)
        performance_ys = evaluate_performance_thresholds(student_simulator, tutor_simulator, prints=prints, CONSTANTS=CONSTANTS)
        performances_ys.append(performance_ys)
    
    mean_performance_ys = np.mean(performances_ys, axis=0)
    min_performance_ys = np.min(performances_ys, axis=0)
    max_performance_ys = np.max(performances_ys, axis=0)
    std_performance_ys = np.std(performances_ys, axis=0)
    xs = np.arange(len(mean_performance_ys)).tolist()
    print("Evaluated mean performance after " + str(avg_over_runs) + " runs", mean_performance_ys[-1])

    if plots:
        file_name = 'plots/Current_RT_Thresholds/village_' + CONSTANTS['VILLAGE'] + '/Current_RT_Thresholds_' + str(CONSTANTS['MAX_TIMESTEPS']) + 'obs_' + CONSTANTS['STUDENT_MODEL_NAME'] + '_' + CONSTANTS['STUDENT_ID'] + '.png'
        plt.title("Current RT policy after " + str(CONSTANTS['MAX_TIMESTEPS']) + " opportunities using current thresholds for " + CONSTANTS['STUDENT_MODEL_NAME'] + '(' + CONSTANTS['STUDENT_ID'] + ')')
        plt.plot(xs, mean_performance_ys, color='b', label=label1)
        plt.fill_between(xs, mean_performance_ys-std_performance_ys, mean_performance_ys+std_performance_ys, alpha=0.3)
        plt.xlabel('#Opportunities')
        plt.ylabel('Avg P(Know) across skills')
        plt.legend()
        plt.savefig(file_name)
        plt.show()
        
        with open("logs/ppo_logs/current_rt_thresholds.txt", "w") as f:
            for i in range(len(xs)):
                f.write(str(xs[i]) + ',' + str(performance_ys[i]) + '\n')

    return xs, mean_performance_ys

if __name__ == '__main__':

    # village 130: ['5A27001753', '5A27001932', '5A28002555', '5A29000477', '6105000515', '6112001212', '6115000404', '6116002085', 'new_student']
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    args = arg_parser()

    student_id = CONSTANTS['STUDENT_ID']
    area_rotation = CONSTANTS['AREA_ROTATION']
    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=student_id, prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])
    set_action_constants(args.type, tutor_simulator)
    state_size  = CONSTANTS["STATE_SIZE"]
    action_size = CONSTANTS["ACTION_SIZE"]

    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(student_id)

    checkpoint_file_name_start = student_id + '~' + args.student_model_name + '~village_' + args.village_num + '~type_' + str(args.type) + '~'
    checkpoint_files = os.listdir('./checkpoints')
    checkpoint_file_name = ""

    for file in checkpoint_files:
        if file[:len(checkpoint_file_name_start)] == checkpoint_file_name_start:
            checkpoint_file_name = file
            break

    new_student_model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], fc2_dims=CONSTANTS['FC2_DIMS'], n_actions=action_size, type=args.type)
    new_student_model.load_state_dict(torch.load("checkpoints/" + checkpoint_file_name))
    print("Loaded model at checkpoints/" + checkpoint_file_name)
    
    fig = plt.figure(figsize=(15, 11))
    ax = []
    for i in range(len(student_simulator.uniq_student_ids)):
        ax.append(fig.add_subplot(3,3,i+1))
        ax[i].clear()
    
    total_threshold_avgs = 0.0
    total_posterior_avgs = 0.0
    student_spec_total_posterior_avgs = 0.0

    for i in range(0, 8):
        student_num = i
        student_id = student_simulator.uniq_student_ids[i]
        if student_id == 'new_student': CONSTANTS['NEW_STUDENT_PARAMS'] = None
        else: CONSTANTS['NEW_STUDENT_PARAMS'] = student_id

        # Initialise student simulator and environment
        student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=student_id, prints=False)
        env = StudentEnv(student_simulator, action_size, student_id, 1, args.type, prints=False, area_rotation=args.area_rotation, CONSTANTS=CONSTANTS)
        env.checkpoint()
        xs, threshold_ys = evaluate_current_RT_thresholds(plots=False, prints=False)

        # Load student_specific_model if necessary
        if CONSTANTS['STUDENT_SPEC_MODEL']:
            checkpoint_file_name_start = student_id + '~' + args.student_model_name + '~village_' + args.village_num + '~type_' + str(args.type) + '~'
            checkpoint_files = os.listdir('./checkpoints')
            checkpoint_file_name = ""
            for file in checkpoint_files:
                if file[:len(checkpoint_file_name_start)] == checkpoint_file_name_start:
                    checkpoint_file_name = file
                    break
            student_spec_model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], fc2_dims=CONSTANTS['FC2_DIMS'], n_actions=action_size, type=args.type)
            student_spec_model.load_state_dict(torch.load("checkpoints/" + checkpoint_file_name))
            print("Loaded model at checkpoints/" + checkpoint_file_name, student_id)

        total_reward, _total_reward = [], []
        posteriors, _posteriors = [], []
        student_avgs, student_spec_avgs = [], []

        for _ in tqdm(range(CONSTANTS['AVG_OVER_RUNS'])):
            env.reset()
            prior = env.state[:22]

            tr, posterior, learning_progress = play_env(env, new_student_model, device, CONSTANTS, deterministic=args.deterministic)
            total_reward.append(tr)
            posteriors.append(posterior)
            student_avg = []
            for know in learning_progress[student_num]:
                avg_know = np.mean(np.array(know))
                student_avg.append(avg_know)
            student_avgs.append(student_avg)

            if CONSTANTS['STUDENT_SPEC_MODEL']:
                _tr, _posterior, _learning_progress = play_env(env, student_spec_model, device, CONSTANTS, deterministic=args.deterministic)
                _total_reward.append(_tr)
                _posteriors.append(_posterior)
                student_spec_avg = []
                for know in _learning_progress[student_num]:
                    _avg_know = np.mean(np.array(know))
                    student_spec_avg.append(_avg_know)
                student_spec_avgs.append(student_spec_avg)

        total_reward = np.mean(total_reward)
        posteriors = np.mean(posteriors, axis=0)
        student_avgs = np.mean(student_avgs, axis=0)
        threshold_policy_improvement = 100 * (np.mean(posteriors) - threshold_ys[-1])/threshold_ys[-1]

        if CONSTANTS['STUDENT_SPEC_MODEL']:
            _total_reward = np.mean(_total_reward)
            _posteriors = np.mean(_posteriors, axis=0)
            student_spec_avgs = np.mean(student_spec_avgs, axis=0)
            student_spec_threshold_policy_improvement = 100 * (np.mean(_posteriors) - threshold_ys[-1])/threshold_ys[-1]
        
        x = np.arange(len(student_avg))
        if CONSTANTS['STUDENT_SPEC_MODEL'] and i != len(student_simulator.uniq_student_ids) - 1: ax[i].plot(x, student_spec_avgs, label="Student-specific RL Policy", color="green", alpha=0.5)
        ax[i].plot(xs, threshold_ys, color='r', label="Current RT Thresholds", alpha=0.5)
        # ax[i].plot(x, student_avg, label="Generic RL Policy Type " + str(args.type), color="black")
        ax[i].set_xlabel("# Opportunities")
        ax[i].set_ylabel("Avg P(Know) across skills")
        ax[i].set_title("Student: " + student_id)
        ax[i].grid()
        ax[i].legend()
        
        print('Student ', i, student_id, "\nnew_student Avg. prior Know: ", np.mean(prior), "\nnew_student Avg. posterior Know: ", np.mean(posteriors))
        if CONSTANTS['STUDENT_SPEC_MODEL'] and i != len(student_simulator.uniq_student_ids) - 1: print("\nspec_student Avg. prior Know: ", np.mean(prior), "\nspec_student Avg. posterior Know: ", np.mean(_posteriors))
        print("threshold averages:", threshold_ys[-1] )

        if CONSTANTS['STUDENT_SPEC_MODEL']:
            print("general policy improvement: ",  threshold_policy_improvement, '\n')
            print("student_spec policy improvement: ",  student_spec_threshold_policy_improvement, '\n')
        total_threshold_avgs += threshold_ys[-1]

        if CONSTANTS['STUDENT_SPEC_MODEL']: student_spec_total_posterior_avgs += np.mean(_posteriors)
        total_posterior_avgs += np.mean(posteriors)
        
    total_threshold_policy_improvement = 100 * (total_posterior_avgs - total_threshold_avgs)/total_threshold_avgs
    print("TOTAL THRESHOLD IMPROVEMENT (generic policy): ", total_threshold_policy_improvement)
    if CONSTANTS['STUDENT_SPEC_MODEL']:
        student_spec_total_threshold_policy_improvement = 100 * (student_spec_total_posterior_avgs - total_threshold_avgs)/total_threshold_avgs
        print("TOTAL THRESHOLD IMPROVEMENT (student_spec policy): ", student_spec_total_threshold_policy_improvement)
    plt.tight_layout()
    plt.savefig('../RoboTutor-Analysis/plots/Played plots/Type ' + str(args.type) + '/village_' + args.village_num + '~obs_' + args.observations + '~' + args.student_model_name + '.png')
    plt.show()