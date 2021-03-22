import sys
import os
import re
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import math
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

sys.path.append('..')

import torch
from torch.utils.tensorboard import SummaryWriter

from lib.common import mkdir
from lib.multiprocessing_env import SubprocVecEnv

from RL_agents.DQN import DQNAgent
from RL_agents.actor_critic_agent import ActorCriticAgent, ActorCritic

from baselines.random_baseline import RandomBaseline
from baselines.pedagogical_baseline import PedagogicalBaseline

from simulators.student_simulator import StudentSimulator
from simulators.tutor_simulator import TutorSimulator
from environment import StudentEnv

from constants import CONSTANTS
from RL_agents.ppo_helper import *
from helper import *
from reader import *

class Simulator():
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu:0")
        self.args = self.arg_parser()
        self.algo = self.args.algo
        print('Device:', self.device)

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
        parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
        parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
        parser.add_argument("-e", "--num_envs", default=CONSTANTS["NUM_ENVS"], help="Number of observations to train on", type=int)
        parser.add_argument("--episodes", default=CONSTANTS["NUM_EPISODES"], help="Number of training episodes", type=int)
        parser.add_argument('-mt', '--max_timesteps', help="Total questions that will be given to the student/RL agent", type=int, default=CONSTANTS['MAX_TIMESTEPS'])
        parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
        parser.add_argument('-smn', '--student_model_name', default=CONSTANTS['STUDENT_MODEL_NAME'], help="Student model name")
        parser.add_argument('--test_epochs', default=CONSTANTS['TEST_EPOCHS'], help="Test after every x epochs", type=int)
        parser.add_argument('--ppo_steps', help="PPO Steps", default=CONSTANTS['PPO_STEPS'], type=int)
        parser.add_argument('-ar', '--area_rotation', help="Area rotation sequence like L-N-L-S", default=CONSTANTS['AREA_ROTATION'])
        parser.add_argument('-arc', '--area_rotation_constraint', help="Should questions be constrained like lit-num-lit-stories? True/False", default=True)
        parser.add_argument('-tc', '--transition_constraint', help="Should transition be constrained to prev,same,next, next-next? True/False", default=True)
        parser.add_argument('--target', default=CONSTANTS['TARGET_P_KNOW'], type=float)
        parser.add_argument('--anti_rl', default=False)
        parser.add_argument('-m', '--model', help="Model file to load from checkpoints directory, if any")
        parser.add_argument('-nsp', '--new_student_params', default=CONSTANTS['NEW_STUDENT_PARAMS'], help="The model params new_student has to start with; enter student_id")
        parser.add_argument("-c", "--clear_logs", default=CONSTANTS["CLEAR_LOGS"])
        parser.add_argument("--algo", default=CONSTANTS["ALGO"], help="dqn/actor_critic/ppo")
        parser.add_argument("--load", default=CONSTANTS["LOAD"])
        parser.add_argument("--avg_runs", default=CONSTANTS["AVG_OVER_RUNS"], help="Number of runs to average experiments")
        parser.add_argument("-d", "--deterministic", default=CONSTANTS['DETERMINISTIC'], help="enable deterministic actions")
        args = parser.parse_args()
        return args

    def set_constants(self, CONSTANTS):
        args = self.args
        CONSTANTS['NUM_OBS']                    = args.observations
        CONSTANTS['VILLAGE']                    = args.village_num
        CONSTANTS['AGENT_TYPE']                 = args.type
        CONSTANTS['NUM_ENVS']                   = args.num_envs
        CONSTANTS['NUM_EPISODES']               = args.episodes
        CONSTANTS['MAX_TIMESTEPS']              = args.max_timesteps
        CONSTANTS['STUDENT_ID']                 = args.student_id
        CONSTANTS['STUDENT_MODEL_NAME']         = args.student_model_name
        CONSTANTS['TEST_EPOCHS']                = args.test_epochs
        CONSTANTS['PPO_STEPS']                  = args.ppo_steps
        CONSTANTS['AREA_ROTATION']              = args.area_rotation
        CONSTANTS['AREA_ROTATION_CONSTRAINT']   = args.area_rotation_constraint
        CONSTANTS['TRANSITION_CONSTRAINT']      = args.transition_constraint
        CONSTANTS['TARGET_P_KNOW']              = args.target
        CONSTANTS['NEW_STUDENT_PARAMS']         = args.new_student_params
        CONSTANTS['CLEAR_LOGS']                 = args.clear_logs
        CONSTANTS['AVG_OVER_RUNS']              = args.avg_runs
        CONSTANTS['LOAD']                       = args.load
        CONSTANTS['ALGO']                       = args.algo
        CONSTANTS['DETERMINISTIC']              = args.deterministic
        if args.algo.lower() != 'ppo':  CONSTANTS['NUM_ENVS'] = 1

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

        return CONSTANTS

    def set_data_paths(self, CONSTANTS, root_path):
        self.data_path = root_path + '/Data/village_' + CONSTANTS['VILLAGE'] + '/village_' + CONSTANTS['VILLAGE'] + '_step_transac.txt'

    def set_action_constants(self, type, tutor_simulator):
        if self.args.type == 4 or self.args.type == 5 or self.args.type == None:
            num_literacy_acts = len(tutor_simulator.literacy_activities)
            num_math_acts = len(tutor_simulator.math_activities)
            num_story_acts = len(tutor_simulator.story_activities)
            CONSTANTS['ACTION_SIZE'] = [num_literacy_acts, num_math_acts, num_story_acts]
            CONSTANTS['NUM_LITERACY_ACTS'], CONSTANTS['NUM_MATH_ACTS'], CONSTANTS['NUM_STORY_ACTS'] = CONSTANTS['ACTION_SIZE'][0], CONSTANTS['ACTION_SIZE'][1], CONSTANTS['ACTION_SIZE'][2]
            CONSTANTS['LITERACY_ACTS'], CONSTANTS['MATH_ACTS'], CONSTANTS['STORY_ACTS'] = tutor_simulator.literacy_activities, tutor_simulator.math_activities, tutor_simulator.story_activities
            if self.args.type == 5 or self.args.type == None:  
                CONSTANTS['ACTION_SIZE'] = num_literacy_acts + num_math_acts + num_story_acts
        else:
            pass
        return CONSTANTS
    
    def set_target_reward(self, CONSTANTS, init_avg_p_know):
        CONSTANTS["TARGET_REWARD"] = 1000 * (CONSTANTS["TARGET_P_KNOW"] - init_avg_p_know)
        return CONSTANTS
    
    def save_params(self, PATH, agent, CONSTANTS):
        if CONSTANTS['ALGO'].lower() == 'dqn':
            torch.save(agent.Q_eval.state_dict(), PATH)
        
        elif CONSTANTS['ALGO'] == 'actor_critic':
            torch.save(agent.actor_critic.state_dict(), PATH)
        
        elif CONSTANTS['ALGO'].lower() == 'ppo':
            torch.save(agent.state_dict(), PATH)

    def load_params(self, agent, CONSTANTS, manual_algo=None):

        if CONSTANTS['LOAD'] == True:
            
            if manual_algo is None: manual_algo = CONSTANTS['ALGO'].lower()

            if manual_algo == 'ppo':
                file_name = 'ppo_'+CONSTANTS['STUDENT_ID']+'_hotDINA_skill_village' + CONSTANTS['VILLAGE'] + '_type' + str(CONSTANTS['AGENT_TYPE']) 
                file_name_regex = re.compile(file_name)
                files = os.listdir('saved_model_parameters')
                PATH = os.getcwd() + "/saved_model_parameters/" + [s for s in files if file_name_regex.match(s)][0]

            else:
                PATH = os.getcwd() + "/saved_model_parameters/" + manual_algo + "_" + CONSTANTS['STUDENT_ID'] +"_type" + str(CONSTANTS['AGENT_TYPE']) + ".pth"

            if manual_algo == "dqn":
                agent.Q_eval.load_state_dict(torch.load(PATH))
                print("loaded model params at: ", PATH)
                agent.Q_eval.eval()

            elif manual_algo == "actor_critic":
                agent.actor_critic.load_state_dict(torch.load(PATH))
                print("loaded model params at: ", PATH)
                agent.actor_critic.eval()
            
            elif manual_algo == "ppo":
                agent.load_state_dict(torch.load(PATH))
                print("loaded model params at: ", PATH)

        return agent

    def run_pedagogical_expert_baseline(self, CONSTANTS, agent_type, student_params, plots=True, prints=True, avg_over_runs=100):
        pedagogical_expert_baseline = PedagogicalBaseline(CONSTANTS)
        learning_progress_means, learning_progress_min, learning_progress_max = pedagogical_expert_baseline.run(CONSTANTS['AVG_OVER_RUNS'], agent_type, student_params)
        return learning_progress_means, learning_progress_min, learning_progress_max

    def run_random_baseline(self, CONSTANTS, student_id):
        random_baseline = RandomBaseline(CONSTANTS, student_id)
        learning_progress_means, learning_progress_min, learning_progress_max = random_baseline.run(CONSTANTS['AVG_OVER_RUNS'], CONSTANTS['AGENT_TYPE'], student_id)
        return learning_progress_means, learning_progress_min, learning_progress_max

    def get_init_know(self, student_simulator, student_id):
        student_model_name = student_simulator.student_model_name
        student_model = student_simulator.student_model
        student_num = student_simulator.uniq_student_ids.index(student_id)

        if student_model_name == 'ActivityBKT':     return np.array(student_model.know[student_num])
        elif student_model_name == 'hotDINA_skill': return np.array(student_model.alpha[student_num][0])
        else:                                       return None

    def init_agent(self, CONSTANTS, student_simulator, manual_algo=None,skill_groups=None, skill_group_to_activity_map=None, tutorID_to_kc_dict=None, kc_to_tutorID_dict=None, cta_tutor_ids=None, kc_list=None):
        
        if manual_algo is None: 
            manual_algo = CONSTANTS['ALGO'].lower()

        if manual_algo == 'dqn':
            agent = DQNAgent(gamma=CONSTANTS['GAMMA'], 
                            epsilon=CONSTANTS['EPSILON'], 
                            batch_size=CONSTANTS['BATCH_SIZE'], 
                            n_actions=CONSTANTS['ACTION_SIZE'], 
                            input_dims=[CONSTANTS['STATE_SIZE']], 
                            lr=CONSTANTS['DQ_LEARNING_RATE'],
                            agent_type=CONSTANTS['AGENT_TYPE'])
    
        elif manual_algo == 'actor_critic':
            agent = ActorCriticAgent(alpha=CONSTANTS['AC_LEARNING_RATE'], 
                                        input_dims=[CONSTANTS['STATE_SIZE']], 
                                        n_actions=CONSTANTS['ACTION_SIZE'],
                                        gamma=CONSTANTS['GAMMA'],
                                        student_simulator=student_simulator,
                                        layer1_size=128, 
                                        layer2_size=256,
                                        layer3_size=256,
                                        layer4_size=128,
                                        agent_type=CONSTANTS['AGENT_TYPE'])

        elif manual_algo == 'ppo':
            agent = ActorCritic(lr=CONSTANTS['LEARNING_RATE'], 
                                input_dims=[CONSTANTS['STATE_SIZE']], 
                                fc1_dims=CONSTANTS['FC1_DIMS'], 
                                fc2_dims=CONSTANTS['FC2_DIMS'], 
                                n_actions=CONSTANTS['ACTION_SIZE'], 
                                agent_type=CONSTANTS['AGENT_TYPE'])

        return agent
    
    def learn_dqn(self, CONSTANTS, agent, env, PATH):

        scores = []
        eps_history = []    # can be used to see how epsilon changes with each timestep/opportunity
        avg_p_knows = []
        avg_scores = []
        score = 0
        student_num = env.student_simulator.uniq_student_ids.index(env.student_simulator.new_student_params)

        for i in range(CONSTANTS['NUM_EPISODES']):
            score, timesteps = 0, 0
            done = False
            state = env.reset()

            if i > 0 and i % 50 == 0:  self.save_params(PATH, agent, CONSTANTS)

            while done is False:
                timesteps += 1
                CONSTANTS['RUN'] += 1
                action, explore, _ = agent.choose_action(state, explore=True)
                prior_know = state[:CONSTANTS['NUM_SKILLS']]
                next_state, reward, student_response, done, posterior_know = env.step(action, CONSTANTS["MAX_TIMESTEPS"], timesteps)
                agent.store_transition(state, action, reward, next_state, done)
                agent.learn(decrease_eps=True)
                state = next_state.copy()
                score += reward
                gain = np.mean(np.array(posterior_know) - np.array(prior_know)) # mean gain in prior -> posterior knowledge of student
                
                if done:
                    avg_p_know = np.mean(np.array(posterior_know))
                    avg_p_knows.append(avg_p_know)

            print("episode: ", i, ", score: ", score, ", Avg_p_know: ", avg_p_know, ", TIMESTEPS: ", timesteps)
            with open("logs/"+CONSTANTS["ALGO"].lower()+"_logs/scores_type" + str(self.args.type) + ".txt", "a") as f:
                text = str(i) + "," + str(score) + "\n"
                f.write(text)
            scores.append(score)
                
        if CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_skill':
            final_p_know = np.array(env.student_simulator.student_model.alpha[student_num][-1])
            final_avg_p_know = np.mean(final_p_know)

        return posterior_know, avg_p_knows

    def learn_actor_critic(self, CONSTANTS, agent, env, PATH):

        scores = []
        avg_p_knows = []
        score = 0
        student_num = env.student_simulator.uniq_student_ids.index(env.student_simulator.new_student_params)

        for i in range(CONSTANTS['NUM_EPISODES']):
            score, timesteps = 0, 0
            done = False
            state = env.reset()

            if i > 0 and i % 50 == 0:  self.save_params(PATH, agent, CONSTANTS)

            while done is False:
                timesteps += 1
                CONSTANTS['RUN'] += 1
                action, explore, _, _ = agent.choose_action(state, explore=False)
                prior_know = state[:CONSTANTS['NUM_SKILLS']]
                next_state, reward, student_response, done, posterior_know = env.step(action, CONSTANTS["MAX_TIMESTEPS"], timesteps)
                agent.learn(state, reward, next_state, done)
                state = next_state.copy()
                score += reward
                gain = np.mean(np.array(posterior_know) - np.array(prior_know)) # mean gain in prior -> posterior knowledge of student
                
                if done:
                    avg_p_know = np.mean(np.array(posterior_know))
                    avg_p_knows.append(avg_p_know)
            
            print("episode: ", i, ", score: ", score, ", Avg_p_know: ", avg_p_know, ", TIMESTEPS: ", timesteps)
            with open("logs/"+CONSTANTS["ALGO"].lower()+"_logs/scores_type" + str(self.args.type) + ".txt", "a") as f:
                text = str(i) + "," + str(score) + "\n"
                f.write(text)
            scores.append(score)
                
        if CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_skill':
            final_p_know = np.array(env.student_simulator.student_model.alpha[student_num][-1])
            final_avg_p_know = np.mean(final_p_know)

        return posterior_know, avg_p_knows

    def learn_ppo(self, CONSTANTS, agent, env, PATH):
        # village 130: ['5A27001753', '5A27001932', '5A28002555', '5A29000477', '6105000515', '6112001212', '6115000404', '6116002085', 'new_student']
        action_size = CONSTANTS['ACTION_SIZE']
        student_id = CONSTANTS['STUDENT_ID']
        # self.evaluate_current_RT_thresholds(CONSTANTS, plots=True, prints=False, avg_over_runs=CONSTANTS['AVG_OVER_RUNS'])
        
        # clear logs
        clear_files("ppo", CONSTANTS['CLEAR_LOGS'], path='logs/', type=CONSTANTS['AGENT_TYPE'])
        os.system('cd runs && rm -rf * && cd ..')
        writer = SummaryWriter('runs/type' + str(CONSTANTS['AGENT_TYPE']))
        file_name_no_reward = 'ppo_' + student_id + '_' + self.args.student_model_name + "_village" + CONSTANTS['VILLAGE'] + "_type" + str(CONSTANTS['AGENT_TYPE'])
        file_filter_expression = re.compile(file_name_no_reward)

        if CONSTANTS['LOAD']: 
            files = os.listdir('saved_model_parameters/') 
            required_file = [s for s in files if file_filter_expression.match(s)][0]
            PATH = 'saved_model_parameters/' + required_file
            agent = self.load_params(agent, CONSTANTS)
        
        frame_idx = 0
        train_epoch = 0
        best_reward = -1000.0
        early_stop = False

        # Prepare environments
        envs = [make_env(i+1, env.student_simulator, CONSTANTS['STUDENT_ID'], CONSTANTS['ACTION_SIZE'], type=CONSTANTS['AGENT_TYPE'], area_rotation=CONSTANTS['AREA_ROTATION'], CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl) for i in range(CONSTANTS["NUM_ENVS"])]
        envs = SubprocVecEnv(envs)
        envs.checkpoint()
        state = np.array(envs.reset())
        loop = 0
        done = False
        timesteps = 0
        
        while not early_stop:
            loop += 1
            # lists to store training data
            log_probs = []
            critic_values = []
            states = []
            actions = []
            rewards = []
            dones = []

            for _ in range(CONSTANTS["PPO_STEPS"]):
                timesteps += 1
                if isinstance(state, list): state = torch.tensor(state)
                if torch.is_tensor(state) == False or isinstance(state, np.ndarray):    state = torch.FloatTensor(state)
                if state.get_device() != self.device:    state = state.to(self.device)

                policy, critic_value = agent(state)
                action = policy.sample()    # sample action from the policy distribution
                next_state, reward, student_response, done, posterior_know = envs.step(action.cpu().numpy(), [CONSTANTS['MAX_TIMESTEPS']] * CONSTANTS['NUM_ENVS'], timesteps=[timesteps]*CONSTANTS['NUM_ENVS'])
                log_prob = policy.log_prob(action)
                critic_values.append(critic_value)

                rewards.append(torch.Tensor(reward).unsqueeze(1).to(self.device))
                log_probs.append(log_prob)
                if done.any(): timesteps = 0
                dones.append(torch.Tensor(1 - done).unsqueeze(1).to(self.device))
                states.append(state)
                actions.append(action)
                state = next_state.copy()
                frame_idx += 1
            
            _, critic_value_ = agent(torch.Tensor(next_state).to(self.device))
            returns         = compute_gae(critic_value_, rewards, dones, critic_values, CONSTANTS["GAMMA"], CONSTANTS["GAE_LAMBDA"])
            returns         = torch.cat(returns).detach()
            log_probs       = torch.cat(log_probs).detach()
            critic_values   = torch.cat(critic_values).detach()
            states          = torch.cat(states)
            actions         = torch.cat(actions)
            advantage       = normalize(returns - critic_values)

            ppo_update(agent, frame_idx, states, actions, log_probs, returns, advantage, CONSTANTS, type_=CONSTANTS['AGENT_TYPE'], writer=writer)
            train_epoch += 1
            print("UPDATING.... Epoch Num:", train_epoch)

            if train_epoch % CONSTANTS['TEST_EPOCHS'] == 0:
            
                student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=self.args.new_student_params, prints=False)
                env = StudentEnv(student_simulator, action_size, student_id, 1, CONSTANTS['AGENT_TYPE'], prints=False, area_rotation=CONSTANTS['AREA_ROTATION'], CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl)
                env.checkpoint()
            
                if env.type >= 1 and env.type <= 5:
                    test_reward = []
                    final_p_know = []
                    for _ in range(CONSTANTS['NUM_TESTS']):
                        student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=self.args.new_student_params, prints=False)
                        env = StudentEnv(student_simulator, action_size, student_id, 1, CONSTANTS['AGENT_TYPE'], prints=False, area_rotation=CONSTANTS['AREA_ROTATION'], CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl)
                        env.checkpoint()
                        tr, fpk = test_env(env, agent, self.device, CONSTANTS, deterministic=False, writer=writer)
                        test_reward.append(tr)
                        final_p_know.append(fpk)
                    test_reward = np.mean(test_reward)
                    final_p_know = np.mean(final_p_know, axis=0) 

                print('Frame %s. reward: %s' % (frame_idx, test_reward))
                final_avg_p_know = np.mean(final_p_know)

                # Save a checkpoint every time we achieve a best reward
                if best_reward < test_reward:
                    print("Best reward updated: %.3f -> %.3f Target reward: %.3f" % (best_reward, test_reward, CONSTANTS["TARGET_REWARD"]))
                    file_name = file_name_no_reward + "~" + str(test_reward) + '.pth'
                    os.chdir('saved_model_parameters')
                    # get all files starting like file_name_no_reward
                    files = os.listdir('.')
                    files_to_del = [s for s in files if file_filter_expression.match(s)]
                    save = True
                    
                    for file in files_to_del:
                        if float(file[:-4].split('~')[-1]) > test_reward:   save = False
                        else:   os.system('rm -rf ' + file)

                    os.chdir('..')
                    if save:    
                        PATH = os.path.join('.', 'saved_model_parameters', file_name)
                        self.save_params(PATH, agent, CONSTANTS)

                    best_reward = test_reward

                if test_reward > CONSTANTS["TARGET_REWARD"]:    early_stop = True

        pass

    def learn(self, CONSTANTS):
        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = CONSTANTS['STUDENT_ID'], prints = False)
        tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], self.args.area_rotation, type=self.args.type, thresholds=True)
        print("Running simulation for student_id:", self.args.student_id, "Type:", self.args.type)

        CONSTANTS = self.set_action_constants(self.args.type, tutor_simulator)
        uniq_student_ids = student_simulator.uniq_student_ids
        print(uniq_student_ids, CONSTANTS['NUM_OBS'])

        if CONSTANTS['STUDENT_MODEL_NAME'] == 'ActivityBKT':
            data_dict = extract_activity_table(student_simulator.CONSTANTS['PATH_TO_ACTIVITY_TABLE'], student_simulator.kc_list, CONSTANTS["NUM_OBS"], CONSTANTS['STUDENT_ID'])
        
        else:
            data_pickle_fname = os.getcwd() + '/../hotDINA/pickles/data/data' + self.args.village_num + '_' + self.args.observations + '.pickle'
            if os.path.isfile(data_pickle_fname) is False:
                os.chdir('../hotDINA')
                get_data_pickle_command = 'python get_data_for_village_n.py -v ' + self.args.village_num + ' -o ' + self.args.observations
                os.system(get_data_pickle_command)
                os.chdir('../RoboTutor-Analysis')

            data_dict = pd.read_pickle(data_pickle_fname)
        
        # student_simulator.update_on_log_data(data_dict)
        env = StudentEnv(student_simulator, CONSTANTS["ACTION_SIZE"], self.args.student_id, CONSTANTS['NUM_ENVS'], self.args.type, prints=False, area_rotation=self.args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl)
        env.checkpoint()
        env.reset()

        init_p_know = self.get_init_know(student_simulator, self.args.student_id)
        init_avg_p_know = np.mean(init_p_know)

        agent = self.init_agent(CONSTANTS, student_simulator)
        PATH = os.getcwd() + "/saved_model_parameters/" + CONSTANTS['ALGO'].lower() + "_" + CONSTANTS['STUDENT_ID'] +"_type" + str(CONSTANTS['AGENT_TYPE']) + ".pth"
        agent = self.load_params(agent, CONSTANTS)
        
        if self.args.algo.lower() == 'ppo':
            CONSTANTS = self.set_target_reward(CONSTANTS, init_avg_p_know)

        if CONSTANTS['ALGO'].lower() == 'dqn':
            posterior_know, avg_p_knows = self.learn_dqn(CONSTANTS, agent, env, PATH)
            avg_p_know = np.mean(posterior_know)

        elif CONSTANTS['ALGO'] == 'actor_critic':
            posterior_know, avg_p_knows = self.learn_actor_critic(CONSTANTS, agent, env, PATH)
            avg_p_know = np.mean(posterior_know)

        elif CONSTANTS['ALGO'].lower() == 'ppo':
            self.learn_ppo(CONSTANTS, agent, env, PATH)

        print()
        print("Experimented with", CONSTANTS['MAX_TIMESTEPS'], "attempts/timesteps" )
        # print("Init know", init_p_know)
        # print("Final know", posterior_know)
        # print("INITIAL AVERAGE P_KNOW: ", init_avg_p_know)
        # print("FINAL AVERAGE_P_KNOW: ", avg_p_know)
        # print("MIN AVERAGE_P_KNOW: ", min(avg_p_knows))
        # print("MAX AVERAGE_P_KNOW: ", max(avg_p_knows))
        # print("IMPROVEMENT: ", posterior_know - init_avg_p_know)
        return CONSTANTS

    def play_dqn(self, CONSTANTS, student_params=None):
        
        if student_params is None: student_params = 'new_student'
        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = CONSTANTS['STUDENT_ID'], prints = False)
        uniq_student_ids = student_simulator.uniq_student_ids

        total_reward, _total_reward = [], []
        posteriors, _posteriors = [], []
        student_avgs, student_spec_avgs = [], []
        learning_progresses = {}

        for uniq_student_id in uniq_student_ids:
            learning_progresses[uniq_student_id] = []

        student_num = uniq_student_ids.index(student_params)

        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = student_params, prints = False)
        agent = self.init_agent(CONSTANTS, student_simulator, manual_algo='dqn')
        agent = self.load_params(agent, CONSTANTS, manual_algo='dqn')
        env = StudentEnv(student_simulator, CONSTANTS["ACTION_SIZE"], student_params, CONSTANTS['NUM_ENVS'], self.args.type, prints=False, area_rotation=self.args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl)
        env.checkpoint()
        
        for run in tqdm(range(CONSTANTS['AVG_OVER_RUNS']), desc='DQN Policy'):
            score, timesteps = 0, 0
            done = False
            state = env.reset()
            student_avg = []
            while done is False:
                timesteps += 1
                CONSTANTS['RUN'] += 1
                action, _, _ = agent.choose_action(state, explore=False)
                prior_know = state[:CONSTANTS['NUM_SKILLS']]
                next_state, reward, student_response, done, posterior_know = env.step(action, CONSTANTS["MAX_TIMESTEPS"], timesteps, reset_after_done=False)
                agent.store_transition(state, action, reward, next_state, done)
                score += reward
                state = next_state.copy()
                gain = np.mean(np.array(posterior_know) - np.array(prior_know)) # mean gain in prior -> posterior knowledge of student
            
            posterior = state[:CONSTANTS['NUM_SKILLS']]
            posteriors.append(posterior)
            
            if CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_full' or CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_skill':
                learning_progress = env.student_simulator.student_model.alpha
                learning_progresses[student_params].append(learning_progress[student_num])
            
            for know in learning_progress[student_num]:
                avg_know = np.mean(np.array(know))
                student_avg.append(avg_know)
            student_avgs.append(student_avg)

        posteriors = np.mean(np.array(posteriors), axis=0)
        student_avgs = np.mean(student_avgs, axis=0)

        avg_learning_progresses = {}

        for key in learning_progresses:
            if len(learning_progresses[key]) > 0:   avg_learning_progresses[key] = np.mean(learning_progresses[key], axis=2)
        
        learning_progress_means = np.mean(avg_learning_progresses[student_params], axis=0)[:]
        learning_progress_stds = np.std(avg_learning_progresses[student_params], axis=0)[:]
        learning_progress_min = learning_progress_means - learning_progress_stds
        learning_progress_max = learning_progress_means + learning_progress_stds

        return avg_learning_progresses, posteriors, learning_progress_means, learning_progress_min, learning_progress_max

    def play_actor_critic(self, CONSTANTS, student_params = None):
        
        if student_params is None: student_params = 'new_student'
        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = student_params, prints = False)
        uniq_student_ids = student_simulator.uniq_student_ids

        total_reward, _total_reward = [], []
        posteriors, _posteriors = [], []
        student_avgs, student_spec_avgs = [], []
        learning_progresses = {}

        for uniq_student_id in uniq_student_ids:
            learning_progresses[uniq_student_id] = []

        student_num = uniq_student_ids.index(student_params)

        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = student_params, prints = False)
        agent = self.init_agent(CONSTANTS, student_simulator, manual_algo='actor_critic')
        agent = self.load_params(agent, CONSTANTS, manual_algo='actor_critic')
        env = StudentEnv(student_simulator, CONSTANTS["ACTION_SIZE"], student_params, CONSTANTS['NUM_ENVS'], self.args.type, prints=False, area_rotation=self.args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl)
        env.checkpoint()
        
        for run in tqdm(range(CONSTANTS['AVG_OVER_RUNS']), desc='Actor Critic Policy'):
            score, timesteps = 0, 0
            done = False
            state = env.reset()
            student_avg = []
            while done is False:
                timesteps += 1
                CONSTANTS['RUN'] += 1
                action, _, _, _ = agent.choose_action(state)
                prior_know = state[:CONSTANTS['NUM_SKILLS']]
                next_state, reward, student_response, done, posterior_know = env.step(action, CONSTANTS["MAX_TIMESTEPS"], timesteps, reset_after_done=False)
                state = next_state.copy()
                score += reward
                gain = np.mean(np.array(posterior_know) - np.array(prior_know)) # mean gain in prior -> posterior knowledge of student
            
            posterior = state[:CONSTANTS['NUM_SKILLS']]
            posteriors.append(posterior)
            
            if CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_full' or CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_skill':
                learning_progress = env.student_simulator.student_model.alpha
                learning_progresses[student_params].append(learning_progress[student_num])
            
            for know in learning_progress[student_num]:
                avg_know = np.mean(np.array(know))
                student_avg.append(avg_know)
            student_avgs.append(student_avg)
        
        posteriors = np.mean(posteriors, axis=0)
        student_avgs = np.mean(student_avgs, axis=0)

        avg_learning_progresses = {}

        for key in learning_progresses:
            if len(learning_progresses[key]) > 0:   avg_learning_progresses[key] = np.mean(learning_progresses[key], axis=2)
        
        learning_progress_means = np.mean(avg_learning_progresses[student_params], axis=0)[:]
        learning_progress_stds = np.std(avg_learning_progresses[student_params], axis=0)[:]
        learning_progress_min = learning_progress_means - learning_progress_stds
        learning_progress_max = learning_progress_means + learning_progress_stds

        return avg_learning_progresses, posteriors, learning_progress_means, learning_progress_min, learning_progress_max

    def play_ppo(self, CONSTANTS, student_params=None):
        
        if student_params is None: student_params='new_student'
        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = student_params, prints = False)
        uniq_student_ids = student_simulator.uniq_student_ids

        total_reward, _total_reward = [], []
        posteriors, _posteriors = [], []
        student_avgs, student_spec_avgs = [], []
        learning_progresses = {}

        for uniq_student_id in uniq_student_ids:
            learning_progresses[uniq_student_id] = []

        student_num = uniq_student_ids.index(student_params)
        
        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = student_params, prints = False)
        agent = self.init_agent(CONSTANTS, student_simulator, manual_algo='ppo')
        agent = self.load_params(agent, CONSTANTS, manual_algo='ppo')
        env = StudentEnv(student_simulator, CONSTANTS["ACTION_SIZE"], student_params, CONSTANTS['NUM_ENVS'], self.args.type, prints=False, area_rotation=self.args.area_rotation, CONSTANTS=CONSTANTS, anti_rl=self.args.anti_rl)
        env.checkpoint()
       
        for _ in tqdm(range(CONSTANTS['AVG_OVER_RUNS']), desc='PPO policy'):
            env.reset()
            prior = env.state[:CONSTANTS['NUM_SKILLS']]
            tr, posterior, learning_progress = play_env(env, agent, self.device, CONSTANTS, deterministic=self.args.deterministic)
            total_reward.append(tr)
            posteriors.append(posterior)
            student_avg = []
            learning_progresses[student_params].append(learning_progress[student_num])
            
            for know in learning_progress[student_num]:
                avg_know = np.mean(np.array(know))
                student_avg.append(avg_know)
            student_avgs.append(student_avg)
        
        total_reward = np.mean(total_reward)
        posteriors = np.mean(posteriors, axis=0)
        student_avgs = np.mean(student_avgs, axis=0)
        
        avg_learning_progresses = {}

        for key in learning_progresses:
            if len(learning_progresses[key]) > 0:
                avg_learning_progresses[key] = np.mean(learning_progresses[key], axis=2)

        learning_progress_means = np.mean(avg_learning_progresses[student_params], axis=0)[:]
        learning_progress_stds = np.std(avg_learning_progresses[student_params], axis=0)[:]
        learning_progress_min = learning_progress_means - learning_progress_stds
        learning_progress_max = learning_progress_means + learning_progress_stds

        return avg_learning_progresses, posteriors, learning_progress_means, learning_progress_min, learning_progress_max

    def play(self, CONSTANTS):
        
        CONSTANTS['LOAD'] = True
        student_simulator = StudentSimulator(village = CONSTANTS["VILLAGE"], observations = CONSTANTS["NUM_OBS"], student_model_name = CONSTANTS["STUDENT_MODEL_NAME"],new_student_params = CONSTANTS['STUDENT_ID'], prints = False)
        tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], self.args.area_rotation, type=self.args.type, thresholds=True)
        CONSTANTS = self.set_action_constants(self.args.type, tutor_simulator)

        uniq_student_ids = student_simulator.uniq_student_ids
        student_id = CONSTANTS['STUDENT_ID']
        student_num = uniq_student_ids.index(student_id)
        # student_simulator.update_on_log_data(data_dict)

        xs = np.arange(CONSTANTS['MAX_TIMESTEPS']+1).tolist()

        student_params = CONSTANTS['NEW_STUDENT_PARAMS']


        CONSTANTS['NEW_STUDENT_PARAMS'] = student_params
        random_policy_label = "Random policy (baseline)"
        random_policy_means, random_policy_min, random_policy_max = self.run_random_baseline(CONSTANTS, student_params)
        plt.plot(xs, random_policy_means, color='maroon', label=random_policy_label, linestyle='dashed')
        plt.fill_between(xs, random_policy_min, random_policy_max, alpha=0.3, color='maroon')
        
        expert_policy_label = "Pedagogical expert (baseline)"
        expert_policy_means, expert_policy_min, expert_policy_max = self.run_pedagogical_expert_baseline(CONSTANTS, CONSTANTS['AGENT_TYPE'], student_params, plots=True, prints=False, avg_over_runs=CONSTANTS['AVG_OVER_RUNS'])
        plt.plot(xs, expert_policy_means, color='blue', label=expert_policy_label, linestyle='dashed')
        plt.fill_between(xs, expert_policy_min, expert_policy_max, alpha=0.3, color='blue')

        dqn_policy_label = 'Teaching policy (DQN) across ' + str(CONSTANTS['AVG_OVER_RUNS']) + ' runs'
        dqn_avg_learning_progresses, dqn_posteriors, dqn_new_student_learning_progress_means, dqn_new_student_learning_progress_min, dqn_new_student_learning_progress_max = self.play_dqn(CONSTANTS, student_params)
        plt.plot(xs, dqn_new_student_learning_progress_means, color='green', label=dqn_policy_label)
        plt.fill_between(xs, dqn_new_student_learning_progress_min, dqn_new_student_learning_progress_max, alpha=0.3, color='green')

        actor_critic_policy_label = 'Teaching policy (Actor_Critic) across ' + str(CONSTANTS['AVG_OVER_RUNS']) + ' runs'
        actor_critic_avg_learning_progresses, actor_critic_posteriors, actor_critic_new_student_learning_progress_means, actor_critic_new_student_learning_progress_min, actor_critic_new_student_learning_progress_max = self.play_actor_critic(CONSTANTS, student_params)
        plt.plot(xs, actor_critic_new_student_learning_progress_means, color='black', label=actor_critic_policy_label)
        plt.fill_between(xs, actor_critic_new_student_learning_progress_min, actor_critic_new_student_learning_progress_max, alpha=0.3, color='black')

        ppo_policy_label = 'Teaching policy (PPO) across ' + str(CONSTANTS['AVG_OVER_RUNS']) + ' runs'
        ppo_avg_learning_progresses, ppo_posteriors, ppo_new_student_learning_progress_means, ppo_new_student_learning_progress_min, ppo_new_student_learning_progress_max = self.play_ppo(CONSTANTS, student_params)
        plt.plot(xs, ppo_new_student_learning_progress_means, color='red', label=ppo_policy_label)
        plt.fill_between(xs, ppo_new_student_learning_progress_min, ppo_new_student_learning_progress_max, alpha=0.3, color='red')
        
        plt.title("Baseline vs teaching policy learning gains for " + str(CONSTANTS['MAX_TIMESTEPS']) + " timesteps " + '(' + student_params + ')')
        plt.xlabel('#Opportunities')
        plt.ylabel('Avg P(Know) across skills')
        plt.legend()
        plt.savefig('plots/Played plots/Type '+str(CONSTANTS['AGENT_TYPE'])+'/policy_evaluation_' + student_params + '.png')
        plt.show()

        dqn_expert_policy_improvement = 100 * (np.mean(dqn_posteriors) - expert_policy_means[-1])/expert_policy_means[-1]
        actor_critic_expert_policy_improvement = 100 * (np.mean(actor_critic_posteriors) - expert_policy_means[-1])/expert_policy_means[-1]
        ppo_expert_policy_improvement = 100 * (np.mean(ppo_posteriors) - expert_policy_means[-1])/expert_policy_means[-1]

        dqn_random_policy_improvement = 100 * (np.mean(dqn_posteriors) - random_policy_means[-1])/random_policy_means[-1]
        actor_critic_random_policy_improvement = 100 * (np.mean(actor_critic_posteriors) - random_policy_means[-1])/random_policy_means[-1]
        ppo_random_policy_improvement = 100 * (np.mean(ppo_posteriors) - random_policy_means[-1])/random_policy_means[-1]

        print()
        print(random_policy_label, random_policy_means[-1])
        print(expert_policy_label, expert_policy_means[-1])
        print(dqn_policy_label, dqn_new_student_learning_progress_means[-1], dqn_expert_policy_improvement, dqn_random_policy_improvement)
        print(actor_critic_policy_label, actor_critic_new_student_learning_progress_means[-1], actor_critic_expert_policy_improvement, actor_critic_random_policy_improvement)
        print(ppo_policy_label, ppo_new_student_learning_progress_means[-1], ppo_expert_policy_improvement, ppo_random_policy_improvement)
