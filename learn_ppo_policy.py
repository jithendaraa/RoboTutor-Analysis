import os
import argparse
import numpy as np
from pathlib import Path
import pickle
import math

import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.common import mkdir
from lib.multiprocessing_env import SubprocVecEnv

from RL_agents.actor_critic_agent import ActorCritic
from simulators.student_simulator import StudentSimulator
from simulators.tutor_simulator import TutorSimulator
from environment import StudentEnv

from RL_agents.ppo_helper import *
from reader import *
from helper import *

CONSTANTS = {
    'NUM_ENVS'                          :   8,
    'NUM_SKILLS'                        :   22,
    'STATE_SIZE'                        :   22,
    'ACTION_SIZE'                       :   43,
    "TARGET_P_KNOW"                     :   0.97,
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
    'PPO_STEPS'                         :   64, # Must be a multiple of MINI_BATCH_SIZE
    'PPO_EPOCHS'                        :   10,
    'TEST_EPOCHS'                       :   5,
    'NUM_TESTS'                         :   10,
    'GAE_LAMBDA'                        :   0.95,
    "MINI_BATCH_SIZE"                   :   32,
    "MAX_TIMESTEPS"                     :   200,
    'GAMMA'                             :   0.99,
    'GAE_LAMBDA'                        :   0.95,
    'PPO_EPSILON'                       :   0.2,
    "CRITIC_DISCOUNT"                   :   0.5, 
    "ENTROPY_BETA"                      :   0.001,
    "RUN_NUM"                           :   0,
    'PRINT_STUDENT_PARAMS'              :   True,
    'CLEAR_LOGS'                        :   True,
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
    CONSTANTS['NUM_ENVS'] = args.envs
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['AGENT_TYPE'] = args.type
    CONSTANTS['STUDENT_ID'] = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['AREA_ROTATION_CONSTRAINT'] = args.area_rotation_constraint
    CONSTANTS['TRANSITION_CONSTRAINT'] = args.transition_constraint
    CONSTANTS['AREA_ROTATION'] = args.area_rotation
    CONSTANTS['MAX_TIMESTEPS'] = args.max_timesteps
    CONSTANTS['TEST_EPOCHS'] = args.test_epochs
    CONSTANTS['PPO_STEPS'] = max(args.max_timesteps, CONSTANTS['PPO_STEPS'])
    CONSTANTS['NEW_STUDENT_PARAMS'] = args.new_student_params
    CONSTANTS['NUM_ENVS'] = args.num_envs
    CONSTANTS['TARGET_P_KNOW'] = args.target
    CONSTANTS['CLEAR_LOGS'] = args.clear_logs

    if args.type == None:
        CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS']
        CONSTANTS['USES_THRESHOLDS'] = None
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = False

    if args.type == 1:  # State size: number of KC's; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS']
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True
        CONSTANTS['FC1_DIMS'] = 256
    
    elif args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS'] + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256

    elif args.type == 3:
        CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS'] + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 4    # prev, same, next, next_next
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256
        CONSTANTS['LEARNING_RATE'] = 1e-4
    
    elif args.type == 4:
        CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS'] + 1 
        CONSTANTS['ACTION_SIZE'] = None
        CONSTANTS['USES_THRESHOLDS'] = False
        CONSTANTS['TRANSITION_CONSTRAINT'] = False
        CONSTANTS['AREA_ROTATION_CONSTRAINT'] = True
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256
        CONSTANTS['LEARNING_RATE'] = 5e-3
    
    elif args.type == 5:
        CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS']
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
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument("-e", "--envs", default=CONSTANTS["NUM_ENVS"], help="Number of observations to train on", type=int)
    parser.add_argument('-mt', '--max_timesteps', help="Total questions that will be given to the student/RL agent", type=int, default=CONSTANTS['MAX_TIMESTEPS'])
    parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
    parser.add_argument('-smn', '--student_model_name', help="Student model name")
    parser.add_argument('--test_epochs', default=CONSTANTS['TEST_EPOCHS'], help="Test after every x epochs", type=int)
    parser.add_argument('--ppo_steps', help="PPO Steps", default=CONSTANTS['PPO_STEPS'], type=int)
    parser.add_argument('-ar', '--area_rotation', help="Area rotation sequence like L-N-L-S", default=CONSTANTS['AREA_ROTATION'])
    parser.add_argument('-arc', '--area_rotation_constraint', help="Should questions be constrained like lit-num-lit-stories? True/False", default=True)
    parser.add_argument('-tc', '--transition_constraint', help="Should transition be constrained to prev,same,next, next-next? True/False", default=True)
    parser.add_argument('--target', default=CONSTANTS['TARGET_P_KNOW'], type=float)
    parser.add_argument('--anti_rl', default=False)
    parser.add_argument('-m', '--model', help="Model file to load from checkpoints directory, if any")
    parser.add_argument('-nsp', '--new_student_params', help="The model params new_student has to start with; enter student_id")
    parser.add_argument("-c", "--clear_logs", default=CONSTANTS["CLEAR_LOGS"])
    args = parser.parse_args()
    set_constants(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0") # -o 200 -mt 200
    print('Device:', device)        # Autodetect CUDA
    return args

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
    print("Evaluated mean performance after" + str(avg_over_runs) + " runs", mean_performance_ys[-1])

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

def set_target_reward(env):
    init_p_know = env.reset()[:CONSTANTS['NUM_SKILLS']]
    init_avg_p_know = np.mean(np.array(init_p_know))
    target_avg_p_know = CONSTANTS["TARGET_P_KNOW"]
    CONSTANTS["TARGET_REWARD"] = 1000 * (target_avg_p_know - init_avg_p_know)
    return init_p_know

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


if __name__ == '__main__':
    
    # village 130: ['5A27001753', '5A27001932', '5A28002555', '5A29000477', '6105000515', '6112001212', '6115000404', '6116002085', 'new_student']
    
    mkdir('.', 'checkpoints')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    args = arg_parser()
    student_id = CONSTANTS['STUDENT_ID']
    area_rotation = CONSTANTS['AREA_ROTATION']

    evaluate_current_RT_thresholds(plots=True, prints=False, avg_over_runs=CONSTANTS['AVG_OVER_RUNS'])

    # clear logs
    clear_files("ppo", args.clear_logs, path='RL_agents/', type=args.type)
    os.chdir('runs')
    os.system('rm -rf *')
    os.chdir('..')
    writer = SummaryWriter('runs/type' + str(args.type))

    # Initialise simulators
    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=args.new_student_params, prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])
    set_action_constants(args.type, tutor_simulator)

    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(student_id)
    state_size  = CONSTANTS["STATE_SIZE"]
    action_size = CONSTANTS["ACTION_SIZE"]

    # Initialise environment and model
    env = StudentEnv(student_simulator, action_size, student_id, 1, args.type, prints=False, area_rotation=args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=args.anti_rl)
    env.checkpoint()
    model = ActorCritic(lr=CONSTANTS["LEARNING_RATE"], input_dims=[state_size], fc1_dims=CONSTANTS["FC1_DIMS"], fc2_dims=CONSTANTS['FC2_DIMS'], n_actions=action_size, type=args.type)
    if args.model != None:  model.load_state_dict(torch.load("checkpoints/"+args.model))
    init_p_know = set_target_reward(env)
    final_p_know = init_p_know.copy()
    final_avg_p_know = np.mean(final_p_know)

    frame_idx = 0
    train_epoch = 0
    best_reward = -1000.0
    early_stop = False

    # Prepare environments
    envs = [make_env(i+1, student_simulator, student_id, action_size, type=args.type, area_rotation=args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=args.anti_rl) for i in range(CONSTANTS["NUM_ENVS"])]
    envs = SubprocVecEnv(envs)
    envs.checkpoint()
    state = np.array(envs.reset())
    loop = 0
    done = False

    while not early_stop:
        loop += 1
        # lists to store training data
        log_probs = []
        critic_values = []
        states = []
        actions = []
        rewards = []
        dones = []
        timesteps = 0

        for _ in range(CONSTANTS["PPO_STEPS"]):
            timesteps += 1
            if isinstance(state, list): state = torch.tensor(state)
            if torch.is_tensor(state) == False or isinstance(state, np.ndarray):    state = torch.FloatTensor(state)
            if state.get_device() != device:    state = state.to(device)

            if args.type == 4:
                policies, values = model(state)
                activity_actions = []
                action = []
                for policy in policies:
                    action.append(policy.sample().cpu().numpy()[0])
                action = torch.tensor(action).to(device)

                # For state in every env
                for i in range(len(state)):
                    row = state[i]
                    matrix_num = int(row[-1].item())
                    if matrix_num == 1: # literacy
                        act = tutor_simulator.literacy_activities[action.cpu().numpy()[i]]
                    elif matrix_num == 2:   # math
                        act = tutor_simulator.math_activities[action.cpu().numpy()[i]]
                    elif matrix_num == 3:   # story
                        act = tutor_simulator.story_activities[action.cpu().numpy()[i]]
                    
                    act_idx = uniq_activities.index(act)
                    activity_actions.append(act_idx)
                
                log_prob = []
                activity_actions = np.array(activity_actions)
                next_state, reward, student_response, done, posterior_know = envs.step(activity_actions, [CONSTANTS['MAX_TIMESTEPS']] * CONSTANTS['NUM_ENVS'], timesteps=[timesteps]*CONSTANTS['NUM_ENVS'])
                
                for i in range(len(policies)):
                    policy = policies[i]
                    lp = policy.log_prob(action[i:i+1])
                    if len(log_prob) == 0:  log_prob = lp
                    else:   log_prob = torch.cat((log_prob, lp), 0)

                critic_values.append(values)

            else:
                policy, critic_value = model(state)
                action = policy.sample()    # sample action from the policy distribution
                next_state, reward, student_response, done, posterior_know = envs.step(action.cpu().numpy(), [CONSTANTS['MAX_TIMESTEPS']] * CONSTANTS['NUM_ENVS'], timesteps=[timesteps]*CONSTANTS['NUM_ENVS'])
                log_prob = policy.log_prob(action)
                critic_values.append(critic_value)
            
            rewards.append(torch.Tensor(reward).unsqueeze(1).to(device))
            log_probs.append(log_prob)
            dones.append(torch.Tensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            actions.append(action)
            state = next_state.copy()
            frame_idx += 1

        _, critic_value_ = model(torch.Tensor(next_state).to(device))
        returns         = compute_gae(critic_value_, rewards, dones, critic_values, CONSTANTS["GAMMA"], CONSTANTS["GAE_LAMBDA"])
        returns         = torch.cat(returns).detach()
        log_probs       = torch.cat(log_probs).detach()
        critic_values   = torch.cat(critic_values).detach()
        states          = torch.cat(states)
        actions         = torch.cat(actions)
        advantage       = normalize(returns - critic_values)

        # According to PPO paper: (states, actions, log_probs, returns, advantage) is together referred to as a "trajectory"
        ppo_update(model, frame_idx, states, actions, log_probs, returns, advantage, CONSTANTS, type_=args.type, writer=writer)
        train_epoch += 1
        print("UPDATING.... Epoch Num:", train_epoch)

        if train_epoch % CONSTANTS['TEST_EPOCHS'] == 0:
            
            student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=args.new_student_params, prints=False)
            env = StudentEnv(student_simulator, action_size, student_id, 1, args.type, prints=False, area_rotation=args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=args.anti_rl)
            env.checkpoint()
            
            if env.type >= 1 and env.type <= 5:
                test_reward = []
                final_p_know = []
                for _ in range(CONSTANTS['NUM_TESTS']):
                    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=args.new_student_params, prints=False)
                    env = StudentEnv(student_simulator, action_size, student_id, 1, args.type, prints=False, area_rotation=args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=args.anti_rl)
                    env.checkpoint()
                    tr, fpk = test_env(env, model, device, CONSTANTS, deterministic=False, writer=writer)
                    test_reward.append(tr)
                    final_p_know.append(fpk)
                test_reward = np.mean(test_reward)
                final_p_know = np.mean(final_p_know, axis=0) 
            
            print('Frame %s. reward: %s' % (frame_idx, test_reward))
            final_avg_p_know = np.mean(final_p_know)
            
            # Save a checkpoint every time we achieve a best reward
            if best_reward < test_reward:
                print("Best reward updated: %.3f -> %.3f Target reward: %.3f" % (best_reward, test_reward, CONSTANTS["TARGET_REWARD"]))
                file_name_no_reward = CONSTANTS['STUDENT_ID'] + '~' + args.student_model_name + "~village_" + args.village_num + "~type_" + str(args.type)
                file_name = file_name_no_reward + "~" + str(test_reward) + '.dat'
                os.chdir('checkpoints')
                # get all files starting like file_name_no_reward
                files = os.listdir('.')
                files_to_del = []
                save = True
                for file in files:
                    if file[:len(file_name_no_reward)] == file_name_no_reward:
                        files_to_del.append(file)
                for file in files_to_del:
                    if float(file[:-4].split('~')[-1]) > test_reward:   
                        save = False
                    else:
                        os.system('rm -rf ' + file)
                
                os.chdir('..')
                if save:
                    fname = os.path.join('.', 'checkpoints', file_name)
                    torch.save(model.state_dict(), fname)
                
                best_reward = test_reward

            if test_reward > CONSTANTS["TARGET_REWARD"]:    early_stop = True

    print("\nINIT P(Know): ", init_p_know)
    print("FINAL P(Know): ", final_p_know)
    print("IMPROVEMENT PER SKILL: ", np.array(final_p_know) - np.array(init_p_know))
    print("INIT AVG P(KNOW): ", np.mean(init_p_know))
    print("FINAL AVG P(KNOW): ", final_avg_p_know)
    print("TOTAL RUNS: ", CONSTANTS["RUN_NUM"])


        
