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
    'NUM_OBS'                   :   '50',
    'VILLAGE'                   :   '130',
    'AGENT_TYPE'                :   2,
    'STUDENT_ID'                :   'new_student',
    'AREA_ROTATION'             :   'L-N-L-S',
    'STUDENT_MODEL_NAME'        :   'hotDINA_skill',
    'USES_THRESHOLDS'           :   True,
    'STATE_SIZE'                :   22,
    'ACTION_SIZE'               :   4,
    'LEARNING_RATE'             :   5e-4,
    'DETERMINISTIC'             :   True,
    "PRINT_PARAMS_FOR_STUDENT"  :   False,
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
    parser.add_argument("-d", "--deterministic", default=CONSTANTS['DETERMINISTIC'], help="enable deterministic actions")
    args = parser.parse_args()
    return args

def set_constants(args):
    CONSTANTS['NUM_OBS'] = args.observations
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['AGENT_TYPE'] = args.type
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['DETERMINITIC'] = args.deterministic

    if args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
        CONSTANTS['STATE_SIZE'] = 22 + 1 + 1
        CONSTANTS['ACTION_SIZE'] = 3
        CONSTANTS['USES_THRESHOLDS'] = True
        CONSTANTS['FC1_DIMS'] = 128
        CONSTANTS['FC2_DIMS'] = 256

def set_student_simulator(student_id):
    student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=student_id, prints=False)
    return student_simulator

def set_env(student_simulator, student_id, agent_type):
    env = StudentEnv(student_simulator, CONSTANTS['ACTION_SIZE'], student_id, 1, agent_type, prints=False, area_rotation=CONSTANTS['AREA_ROTATION'], CONSTANTS=CONSTANTS)
    env.checkpoint()
    env.reset()
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

def get_rl_act_name(matrix_num, env):
    tutor_simulator = env.tutor_simulator

    if matrix_num == 0:
        pos = tutor_simulator.literacy_pos
        matrix = tutor_simulator.literacy_matrix
    elif matrix_num == 1:
        pos = tutor_simulator.math_pos
        matrix = tutor_simulator.math_matrix
    elif matrix_num == 3:
        pos = tutor_simulator.stories_pos
        matrix = tutor_simulator.stories_matrix
    
    activity_name = matrix[pos[0]][pos[1]]
    return activity_name

def get_matrix_state(matrix_type, tutor_simulator):

    if matrix_type == 'L':
        matrix_type = [1] 
        pos = tutor_simulator.literacy_pos
        matrix = tutor_simulator.literacy_matrix 
        activities = tutor_simulator.literacy_activities
    elif matrix_type == 'N':
        matrix_type = [2]
        pos = tutor_simulator.math_pos
        matrix = tutor_simulator.math_matrix
        activities = tutor_simulator.math_activities
    elif matrix_type == 'S':
        matrix_type = [3]
        pos = tutor_simulator.stories_pos
        matrix = tutor_simulator.stories_matrix
        activities = tutor_simulator.story_activities
    
    activity_name = matrix[pos[0]][pos[1]]
    posn = activities.index(activity_name)

    return matrix_type + [posn]


def global_policy_evaluation(data, uniq_student_ids):
    
    RT_local_policy_score = 0.0
    RT_local_policy_scores = []

    RL_local_policy_score = 0.0
    RL_local_policy_scores = []

    timesteps = int(CONSTANTS['NUM_OBS'])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")

    data['matrix_types'] = []
    policy_evals = []

    for i in range(timesteps):
        user = data['users'][i]
        student_id = uniq_student_ids[user]
        act_num = data['items'][i]
        act_name = data['activities'][i]

        if i == 0 or data['users'][i-1] != data['users'][i]:
            timesteps = 0
            RT_student_simulator = set_student_simulator(student_id)
            model = get_loaded_model(student_id, args.type)
            RL_student_simulator = set_student_simulator(student_id)

        env = set_env(RL_student_simulator, student_id, args.type)

        if act_name in env.tutor_simulator.literacy_activities:
            matrix_type = 'L'
            matrix_num = 0
        elif act_name in env.tutor_simulator.math_activities:
            matrix_type = 'N'
            matrix_num = 1
        elif act_name in env.tutor_simulator.story_activities:
            matrix_type = 'S'
            matrix_num = 3
        data['matrix_types'].append(matrix_type)
        # print(matrix_type, matrix_num)
        env.set_attempt(matrix_num)
        
        RL_student_simulator.student_model.alpha = RT_student_simulator.student_model.alpha.copy()
        env.set_state(RL_student_simulator.student_model.alpha[user][-1] + get_matrix_state(matrix_type, env.tutor_simulator))
        RT_local_policy_score = RT_local_policy_evaluation(data, i, uniq_student_ids, RT_student_simulator) - np.mean(RT_student_simulator.student_model.alpha[user][-2])
        RT_local_policy_scores.append(RT_local_policy_score)
    
        timesteps += 1

        state = env.state.copy()
        state = torch.Tensor(state).unsqueeze(0).to(device)
        policy, _ = model(state)

        if env.type == 2:
            action = policy.probs.cpu().detach().numpy()[0]
            if CONSTANTS['DETERMINISTIC'] == False:
                action = policy.sample().cpu().numpy()[0]
            next_state, reward, _, done, posterior = env.step(action, int(CONSTANTS['NUM_OBS']), timesteps=timesteps, bayesian_update=True, reset_after_done=False)
            print(len(RL_student_simulator.student_model.alpha[student_num]))
        rl_act_name = get_rl_act_name(matrix_num, env)

        RL_local_policy_score = np.mean(posterior) - np.mean(RL_student_simulator.student_model.alpha[student_num][-1])
        RL_local_policy_scores.append(RL_local_policy_score)

        print("RT Activity Name: ", act_name, "RL Activity Name: ",rl_act_name)
        print("RT Local policy gain: ", RT_local_policy_score, "RL Local policy gain: ", RL_local_policy_score)
        policy_evals.append([act_name, RT_local_policy_score, rl_act_name, RL_local_policy_score])

    RT_global_policy_score = np.sum(RT_local_policy_scores)
    RL_global_policy_score = np.sum(RL_local_policy_scores)

    print("RT Global policy score (RT): ", RT_global_policy_score)
    print("RL Global policy score (RL): ", RL_global_policy_score)

    policy_evals = np.array(policy_evals)

    return RT_global_policy_score, RL_global_policy_score, pd.DataFrame(policy_evals, columns=['Activity Name (RoboTutor)', 'Local Policy score (RoboTutor)', 'Activity Name (RL Agent)', 'Local Policy Score (RL Agent)'])


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
    _, _, policy_evals = global_policy_evaluation(data, uniq_student_ids)
    
    policy_evals.to_excel("policy_evals.xlsx")
    
    pass