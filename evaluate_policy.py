import argparse
import pickle
import os
import sys
import copy
import tqdm

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
    'AGENT_TYPE'                :   5,
    'STUDENT_ID'                :   'new_student',
    'AREA_ROTATION'             :   'L-N-L-S',
    'STUDENT_MODEL_NAME'        :   'hotDINA_skill',
    'USES_THRESHOLDS'           :   True,
    'STATE_SIZE'                :   22,
    'ACTION_SIZE'               :   4,
    'LEARNING_RATE'             :   5e-4,
    "MAX_TIMESTEPS"             :   100,
    'DETERMINISTIC'             :   False,
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
    parser.add_argument('-mt', '--max_timesteps', help="Total questions that will be given to the student/RL agent", type=int, default=CONSTANTS['MAX_TIMESTEPS'])
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
    CONSTANTS['DETERMINISTIC'] = args.deterministic
    CONSTANTS['MAX_TIMESTEPS'] = args.max_timesteps

    if args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
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
        CONSTANTS['LEARNING_RATE'] = 1e-4
    
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

def run_historical_attempts(data, i, uniq_student_ids, student_simulator):
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

def get_matrix_type_and_num(env, act_name):
    if act_name in env.tutor_simulator.literacy_activities:
        matrix_type = 'L'
        matrix_num = 0
    elif act_name in env.tutor_simulator.math_activities:
        matrix_type = 'N'
        matrix_num = 1
    elif act_name in env.tutor_simulator.story_activities:
        matrix_type = 'S'
        matrix_num = 3
    else:
        print(act_name, ": NOT FOUND")
    return matrix_type, matrix_num

def get_data_dict(args):
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
    data['matrix_type'] = []
    return data

def local_policy_evaluation(data, uniq_student_ids):
    
    RT_local_policy_score = 0.0
    RT_local_policy_scores = []
    RL_local_policy_score = 0.0
    RL_local_policy_scores = []
    max_timesteps = int(CONSTANTS['MAX_TIMESTEPS'])
    timesteps = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    data['matrix_types'] = []
    policy_evals = []

    for i in tqdm(range(len(data['users']))):
        timesteps += 1
        user = data['users'][i]
        student_id = uniq_student_ids[user]
        act_num = data['items'][i]
        underscored_act_name = data['activities'][i]

        if i == 0 or data['users'][i-1] != data['users'][i]:
            timesteps = 1
            model = get_loaded_model(student_id, CONSTANTS['AGENT_TYPE'])
            RT_student_simulator = set_student_simulator(student_id)
            RL_student_simulator = set_student_simulator(student_id)
            underscore_to_colon_tutor_id_dict = student_simulator.underscore_to_colon_tutor_id_dict
        elif timesteps > max_timesteps:
            continue
        
        colon_act_name = underscore_to_colon_tutor_id_dict[underscored_act_name]
        env = set_env(RL_student_simulator, student_id, args.type)
        matrix_type, matrix_num = get_matrix_type_and_num(env, colon_act_name)
        data['matrix_types'].append(matrix_type)
        env.set_attempt(matrix_num)
        
        RL_student_simulator.student_model.alpha = copy.deepcopy(RT_student_simulator.student_model.alpha)
        if args.type == 2 or args.type == 3:    env.set_state(RL_student_simulator.student_model.alpha[user][-1] + get_matrix_state(matrix_type, env.tutor_simulator))
        elif args.type == 4:                    env.set_state(RL_student_simulator.student_model.alpha[user][-1] + get_matrix_state(matrix_type, env.tutor_simulator)[:1])
        elif args.type == 5:                    env.set_state(RL_student_simulator.student_model.alpha[user][-1])
        RT_local_policy_score = run_historical_attempts(data, i, uniq_student_ids, RT_student_simulator) - np.mean(RT_student_simulator.student_model.alpha[user][-2])
        RT_local_policy_scores.append(RT_local_policy_score)
        # print("PRIOR (RT Local): ", np.mean(RT_student_simulator.student_model.alpha[user][-2]), " --> POSTERIOR (RT Local): ", np.mean(RT_student_simulator.student_model.alpha[user][-1]), "RT Gain: [", np.mean(RT_student_simulator.student_model.alpha[user][-1]) - np.mean(RT_student_simulator.student_model.alpha[user][-2]) ,"]", colon_act_name, data['y'][i])
        skill_group_nums = data['skill_group_nums'][i]
        skill_group_names = data['skill_group_names'][i]
        priors, posteriors = [], []
        for skill in skill_group_nums:
            priors.append(RT_student_simulator.student_model.alpha[user][-2][skill])
            posteriors.append(RT_student_simulator.student_model.alpha[user][-1][skill])
        # print(skill_group_names, priors, posteriors)
        # print("PRIOR FULL: ", RT_student_simulator.student_model.alpha[user][-2])
        # print("POSTERIOR FULL: ", RT_student_simulator.student_model.alpha[user][-1])
        state = env.state.copy()
        state = torch.Tensor(state).unsqueeze(0).to(device)
        policy, _ = model(state)

        if env.type == 2:
            action = policy.probs.cpu().detach().numpy()[0]
            if CONSTANTS['DETERMINISTIC'] == False:
                action = policy.sample().cpu().numpy()[0]
            next_state, reward, _, done, posterior = env.step(action, max_timesteps, timesteps=timesteps, bayesian_update=True, reset_after_done=False)
            rl_act_name = get_rl_act_name(matrix_num, env)
            # print("PRIOR (RL Local)", np.mean(RL_student_simulator.student_model.alpha[user][-2]), "--> POSTERIOR (RL Local)", np.mean(posterior), "RT Gain: [", np.mean(posterior) - np.mean(RT_student_simulator.student_model.alpha[user][-2]) ,"]", rl_act_name, _)
        
        elif env.type == 3 or env.type == 5:
            if CONSTANTS['DETERMINISTIC'] == False:
                action = policy.sample().cpu().numpy()[0]
            else:
                action = policy.probs.cpu().detach().numpy()[0]
                action = action.tolist().index(max(action))
            next_state, reward, _, done, posterior = env.step(action, max_timesteps, timesteps=timesteps, bayesian_update=True, reset_after_done=False)
            if env.type == 3:   rl_act_name = get_rl_act_name(matrix_num, env)
            if env.type == 5:   rl_act_name = RL_student_simulator.uniq_activities[action]
            # print("PRIOR (RL Local)", np.mean(RL_student_simulator.student_model.alpha[user][-2]), "--> POSTERIOR (RL Local)", np.mean(posterior), "RT Gain: [", np.mean(posterior) - np.mean(RT_student_simulator.student_model.alpha[user][-2]) ,"]", rl_act_name, _)
        
        elif env.type == 4:
            policy = policy[0]
            row = state[0]
            if CONSTANTS['DETERMINISTIC'] == False:  action = policy.sample().cpu().numpy()[0]
            else:
                action = policy.probs.cpu().detach().numpy()[0]
                action = action.tolist().index(max(action))

            matrix_num = int(row[-1].item())
            if matrix_num == 1:     rl_act_name = CONSTANTS['LITERACY_ACTS'][action]
            elif matrix_num == 2:   rl_act_name = CONSTANTS['MATH_ACTS'][action]
            elif matrix_num == 3:   rl_act_name = CONSTANTS['STORY_ACTS'][action]
            act_num = env.student_simulator.uniq_activities.index(rl_act_name)
            next_state, reward, _, done, posterior = env.step(act_num, CONSTANTS["MAX_TIMESTEPS"], timesteps=timesteps, bayesian_update=True, reset_after_done=False)

        rl_involved_skills = []
        for p in range(len(posterior)):
            if posterior[p] != RT_student_simulator.student_model.alpha[user][-2][p]: rl_involved_skills.append(p)
        RL_local_policy_score = np.mean(posterior) - np.mean(RL_student_simulator.student_model.alpha[user][-2])
        RL_local_policy_scores.append(RL_local_policy_score)
        policy_evals.append([student_id, 
                            colon_act_name, 
                            RT_local_policy_score, 
                            np.mean(RT_student_simulator.student_model.alpha[user][-2]),
                            np.mean(RT_student_simulator.student_model.alpha[user][-1]),
                            ','.join(str(x) for x in RT_student_simulator.student_model.alpha[user][-2]), 
                            ','.join(str(x) for x in RT_student_simulator.student_model.alpha[user][-1]), 
                            ','.join(str(x) for x in skill_group_nums), 
                            data['y'][i], 
                            rl_act_name, 
                            rl_involved_skills,
                            np.mean(RL_student_simulator.student_model.alpha[user][-1]),
                            ','.join(str(x) for x in RL_student_simulator.student_model.alpha[user][-1]), 
                            RL_local_policy_score, 
                            _])

    policy_evals = np.array(policy_evals)
    RT_total_local_policy_score = np.mean(RT_local_policy_scores)
    RL_total_local_policy_score = np.mean(RL_local_policy_scores)
    
    return RT_total_local_policy_score, RL_total_local_policy_score, pd.DataFrame(policy_evals, columns=['Student ID', 
                                                                                                        'Activity Name (Historical)', 
                                                                                                        'Policy score (Historical)', 
                                                                                                        'Avg. Prior (Historical)',
                                                                                                        'Avg. Posterior (Historical)',
                                                                                                        'Prior (Historical)', 
                                                                                                        'Posterior (Historical)', 
                                                                                                        'Involved skills (Historical)', 
                                                                                                        'Historical Response', 
                                                                                                        'Activity Name (Local)', 
                                                                                                        'Involved Skills (Local)',
                                                                                                        'Avg. Posterior (Local)',
                                                                                                        'Posterior (Local)', 
                                                                                                        'Local Policy Score', 
                                                                                                        'Simulated Response (Local)'])

def global_policy_evaluation(data, uniq_student_ids):
    
    RT_global_policy_score = 0.0
    RT_global_policy_scores = []
    RL_global_policy_score = 0.0
    RL_global_policy_scores = []
    max_timesteps = int(CONSTANTS['MAX_TIMESTEPS'])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu:0")
    data['matrix_types'] = []
    policy_evals = []
    timesteps = 0

    for i in range(len(data['users'])):
        timesteps += 1
        
        user = data['users'][i]
        student_id = uniq_student_ids[user]
        act_num = data['items'][i]
        underscored_act_name = data['activities'][i]
        skill_group_nums = data['skill_group_nums'][i]
        skill_group_names = data['skill_group_names'][i]

        if i == 0 or data['users'][i-1] != data['users'][i]:
            timesteps = 1
            model = get_loaded_model(student_id, args.type)
            RL_student_simulator = set_student_simulator(student_id)
            env = set_env(RL_student_simulator, student_id, args.type)
            knows = [RL_student_simulator.student_model.alpha[user][0]]
            underscore_to_colon_tutor_id_dict = student_simulator.underscore_to_colon_tutor_id_dict
        
        elif timesteps > max_timesteps: continue

        colon_act_name = underscore_to_colon_tutor_id_dict[underscored_act_name]
        state = env.state.copy()
        state = torch.Tensor(state).unsqueeze(0).to(device)
        policy, _ = model(state)
        
        if env.type == 2:
            matrix_type, matrix_num = get_matrix_type_and_num(env, colon_act_name)
            action = policy.probs.cpu().detach().numpy()[0]
            if CONSTANTS['DETERMINISTIC'] == False:
                action = policy.sample().cpu().numpy()[0]
            next_state, reward, _, done, posterior = env.step(action, max_timesteps, timesteps=timesteps, bayesian_update=True, reset_after_done=False)
            rl_act_name = get_rl_act_name(matrix_num, env)
        
        elif env.type == 3 or env.type == 5:
            if CONSTANTS['DETERMINISTIC'] == False:
                action = policy.sample().cpu().numpy()[0]
            else:
                action = policy.probs.cpu().detach().numpy()[0]
                action = action.tolist().index(max(action))
            next_state, reward, _, done, posterior = env.step(action, max_timesteps, timesteps=timesteps, bayesian_update=True, reset_after_done=False)
            if env.type == 3:   
                matrix_type, matrix_num = get_matrix_type_and_num(env, colon_act_name)
                rl_act_name = get_rl_act_name(matrix_num, env)
            elif env.type == 5: rl_act_name = RL_student_simulator.uniq_activities[action]
        
        elif env.type == 4:
            policy = policy[0]
            row = state[0]
            if CONSTANTS['DETERMINISTIC'] == False:  action = policy.sample().cpu().numpy()[0]
            else:
                action = policy.probs.cpu().detach().numpy()[0]
                action = action.tolist().index(max(action))

            matrix_num = int(row[-1].item())
            if matrix_num == 1:     rl_act_name = CONSTANTS['LITERACY_ACTS'][action]
            elif matrix_num == 2:   rl_act_name = CONSTANTS['MATH_ACTS'][action]
            elif matrix_num == 3:   rl_act_name = CONSTANTS['STORY_ACTS'][action]
            act_num = env.student_simulator.uniq_activities.index(rl_act_name)
            next_state, reward, _, done, posterior = env.step(act_num, CONSTANTS["MAX_TIMESTEPS"], timesteps=timesteps, bayesian_update=True, reset_after_done=False)

        rl_involved_skills = []
        for p in range(len(posterior)):
            if posterior[p] != RL_student_simulator.student_model.alpha[user][-2][p]:  rl_involved_skills.append(p)
        RL_global_policy_score = np.mean(posterior) - np.mean(RL_student_simulator.student_model.alpha[user][-2])
        RL_global_policy_scores.append(RL_global_policy_score)
        policy_evals.append([rl_act_name, 
                            RL_global_policy_score, 
                            rl_involved_skills, 
                            np.mean(RL_student_simulator.student_model.alpha[user][-2]),
                            np.mean(RL_student_simulator.student_model.alpha[user][-1]),
                            ",".join(str(x) for x in RL_student_simulator.student_model.alpha[user][-2]), 
                            ",".join(str(x) for x in RL_student_simulator.student_model.alpha[user][-1]), 
                            _])

    policy_evals = np.array(policy_evals)
    RL_total_global_policy_score = np.mean(RL_global_policy_scores)
    # print("RL Total Global policy score :", RL_total_global_policy_score)
    
    return 0, RL_total_global_policy_score, pd.DataFrame(policy_evals, columns=['Activity Name (Global)',
                                                                                'Global Policy Score', 
                                                                                "Involved skills (Global)",
                                                                                "Avg. Prior (Global)", 
                                                                                "Avg. Posterior (Global)", 
                                                                                "Prior (Global)", 
                                                                                "Posterior (Global)", 
                                                                                'Simulated Response (Global)'])

if __name__ == '__main__':
    
    args = arg_parser()
    set_constants(args)
    AVG_OVER_RUNS = 1
    student_simulator = StudentSimulator(village=args.village_num, observations=args.observations, student_model_name=args.student_model_name, new_student_params=CONSTANTS['STUDENT_ID'], prints=False)
    tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], area_rotation=CONSTANTS['AREA_ROTATION'], type=CONSTANTS['AGENT_TYPE'], thresholds=CONSTANTS['USES_THRESHOLDS'])
    set_action_constants(args.type, tutor_simulator)
    state_size  = CONSTANTS["STATE_SIZE"]
    action_size = CONSTANTS["ACTION_SIZE"]
    student_id = CONSTANTS['STUDENT_ID']
    uniq_activities = student_simulator.uniq_activities
    uniq_student_ids = student_simulator.uniq_student_ids
    student_num = uniq_student_ids.index(student_id)
    data_dict = get_data_dict(args)
    RT_locals, RL_locals, RT_globals, RL_globals = [], [], [], []

    for i in (range(AVG_OVER_RUNS)): 
        RT_local, RL_local, local_policy_evals = local_policy_evaluation(data_dict, uniq_student_ids)
        RT_global, RL_global, global_policy_evals = global_policy_evaluation(data_dict, uniq_student_ids)

        RT_locals.append(RT_local) 
        RL_locals.append(RL_local) 
        RL_globals.append(RL_global) 

    RT_locals = np.mean(RT_locals)
    RL_locals = np.mean(RL_locals)
    RL_globals = np.mean(RL_globals)

    print("RT Eval: ", RT_locals)
    print("RL Local Eval: ", RL_locals)
    print("RL Global Eval: ", RL_globals)

    result = pd.concat([local_policy_evals, global_policy_evals], axis=1)
    result.to_excel('policy_evals_type' + str(args.type - 1) + '.xlsx')