import numpy as np
import pandas as pd
import torch
import argparse
import pickle

from helper import *
from reader import *

# RL Agents
from RL_agents.DQN import DQNAgent
# from RL_agents.actor_critic_agent import ActorCriticAgent

# Environment and Student Simulator
from environment import StudentEnv
from simulators.student_simulator import StudentSimulator
from simulators.tutor_simulator import TutorSimulator
from constants import CONSTANTS


def set_constants(args):
    CONSTANTS['NUM_OBS']                    = args.observations
    CONSTANTS['VILLAGE']                    = args.village_num
    CONSTANTS['AGENT_TYPE']                 = args.type
    CONSTANTS['NUM_ENVS']                   = args.num_envs
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

def get_data_dict(matrix_type, student_id, student_simulator):
    
    if CONSTANTS['STUDENT_MODEL_NAME'] == 'ActivityBKT':
        if matrix_type != None:
            activity_df = student_simulator.activity_df[student_simulator.activity_df["Matrix_ActivityName"] == matrix_type]

        else:
            activity_df = student_simulator.activity_df

        student_1_act_df = activity_df

        if student_id != None:
            student_1_act_df = activity_df[activity_df['Unique_Child_ID_1'] == student_id]

        data_dict = extract_activity_table(student_1_act_df, 
                                            student_simulator.act_student_id_to_number_map, 
                                            student_simulator.kc_list)
    
    return data_dict

def get_init_know(student_simulator, student_id):

    student_model_name = student_simulator.student_model_name
    student_model = student_simulator.student_model
    student_num = student_simulator.uniq_student_ids.index(student_id)

    if student_model_name == 'ActivityBKT':
        return np.array(student_model.know[student_num])
        
    elif student_model_name == 'hotDINA_skill':
        # Handle later
        return None

def set_state_size(num_skills):
    if CONSTANTS['ACTION_TYPE'] == 'per_skill_group':
        CONSTANTS['STATE_SIZE'] = num_skills

def init_agent(skill_groups, student_simulator, skill_group_to_activity_map, tutorID_to_kc_dict, kc_to_tutorID_dict, cta_tutor_ids, kc_list):

    if CONSTANTS['ALGO'] == 'actor_critic':
        agent = ActorCriticAgent(alpha=CONSTANTS["LEARNING_RATE"], 
                                    input_dims=[CONSTANTS["STATE_SIZE"]], 
                                    skill_groups=uniq_skill_groups,
                                    skill_group_to_activity_map=skill_group_to_activity_map,
                                    student_simulator=student_simulator,
                                    gamma=CONSTANTS["GAMMA"],
                                    layer1_size=4096, 
                                    layer2_size=2048,
                                    layer3_size=1024,
                                    layer4_size=512,
                                    n_actions=CONSTANTS["ACTION_SIZE"])
    
    elif CONSTANTS["ALGO"] == "dqn":
        agent = DQNAgent(gamma=CONSTANTS['GAMMA'], 
                        epsilon=CONSTANTS['EPSILON'], 
                        batch_size=64, 
                        n_actions=CONSTANTS["ACTION_SIZE"], 
                        input_dims=[CONSTANTS["STATE_SIZE"]], 
                        lr=0.003, 
                        activity_to_skills_map=tutorID_to_kc_dict, 
                        kc_to_tutorID_dict=kc_to_tutorID_dict, 
                        cta_tutor_ids=cta_tutor_ids, 
                        kc_list=kc_list)
    
    return agent

def load_params(PATH, agent):

    if CONSTANTS['LOAD'] == True:
        if CONSTANTS["ALGO"] == "dqn":
            agent.Q_eval.load_state_dict(torch.load(PATH))
            agent.Q_eval.eval()
        elif CONSTANTS["ALGO"] == "actor_critic":
            agent.actor_critic.load_state_dict(torch.load(PATH))
            agent.actor_critic.eval()
    return agent

def save_params(PATH, agent):
    if CONSTANTS["ALGO"] == "dqn":
        torch.save(agent.Q_eval.state_dict(), PATH)
        eps_history.append(agent.epsilon)
    elif CONSTANTS["ALGO"] == "actor_critic":
        torch.save(agent.actor_critic.state_dict(), PATH)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument("-e", "--num_envs", default=CONSTANTS["NUM_ENVS"], help="Number of observations to train on", type=int)
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

if __name__ == '__main__':

    
    args = arg_parser()

    # clear_files(CONSTANTS["ALGO"], CONSTANTS["CLEAR_FILES"])
    # set_constants(args)
    # matrix_type = args.matrix_type
    # student_id = args.student_id

    # student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
    #                                     observations=CONSTANTS["NUM_OBS"], 
    #                                     student_model_name=CONSTANTS["STUDENT_MODEL_NAME"],
    #                                     matrix_type=CONSTANTS['MATRIX_TYPE'])

    # student_num = student_simulator.uniq_student_ids.index(student_id)
    # kc_list = student_simulator.kc_list
    # cta_df = student_simulator.cta_df
    # _, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
    # num_skills = len(kc_list)

    # data_dict = extract_activity_table(student_simulator.uniq_student_ids, kc_list, CONSTANTS['MATRIX_TYPE'], CONSTANTS["NUM_OBS"], CONSTANTS['STUDENT_ID'])
    # student_simulator.update_on_log_data(data_dict)

    # env = StudentEnv(student_simulator=student_simulator,
    #                 skill_groups=uniq_skill_groups,
    #                 skill_group_to_activity_map = skill_group_to_activity_map,
    #                 action_size=CONSTANTS["ACTION_SIZE"],
    #                 student_id=student_id)
    # env.checkpoint()

    # init_p_know = get_init_know(student_simulator, student_id)
    # init_avg_p_know = np.mean(init_p_know)
    # set_state_size(num_skills)

    # agent = init_agent(uniq_skill_groups, student_simulator, skill_group_to_activity_map, tutorID_to_kc_dict, kc_to_tutorID_dict, cta_tutor_ids, kc_list)

    # scores = []
    # eps_history = []    # can be used to see how epsilon changes with each timestep/opportunity
    # score = 0
    # avg_p_knows = []
    # avg_p_know = 0
    # avg_scores = []

    # PATH = "saved_model_parameters/" + CONSTANTS["ALGO"] + ".pth"
    # agent = load_params(PATH, agent)

    # for i in range(CONSTANTS['NUM_EPISODES']):

    #     score = 0
    #     state = env.reset()
    #     done = False
    #     timesteps = 0

    #     output_avg_p_know(i, CONSTANTS["AVG_OVER_EPISODES"], scores, CONSTANTS["ALGO"]+"_logs/avg_scores.txt", avg_p_know)

    #     if i > 0 and i % 500 == 0:
    #         save_params(PATH, agent)

    #     while done != True:
    #         timesteps += 1
    #         CONSTANTS["RUN"] += 1

    #         action, explore, sample_skill, activityName = agent.choose_action(state, explore=False)

    #         prior_know = skill_probas(activityName, tutorID_to_kc_dict, kc_list, state)
    #         next_state, reward, student_response, done, posterior = env.step(action, timesteps, CONSTANTS["MAX_TIMESTEPS"], activityName)
    #         posterior_know = skill_probas(activityName, tutorID_to_kc_dict, kc_list, next_state)

    #         if CONSTANTS["ALGO"] == "dqn":
    #             agent.store_transition(state, action, reward, next_state, done)
    #             agent.learn(decrease_eps=True)

    #         elif CONSTANTS["ALGO"] == "actor_critic":
    #             agent.learn(state, reward, next_state, done)

    #         state = next_state.copy()
    #         score += reward
    #         gain = np.sum(np.array(posterior_know) - np.array(prior_know))/num_skills

    #         with open(CONSTANTS["ALGO"]+"_logs/run.txt", "a") as f:
    #             explore_status = "EXPLOIT"
    #             if explore:
    #                 explore_status = "EXPLORE"
    #             text = "----------------------------------------------------------------------------------------------\n" + "RUN: " + str(CONSTANTS["RUN"]) + "\nTIMESTEPS: " + str(timesteps) + "\n EPISODE: " + str(i) + "\n" + explore_status + "\n" + "Action chosen: " + activityName + "\n" + " Skills: " + str(sample_skill) + "\n" + " Prior P(Know) for these skills: " + str(prior_know) + "\n" + " Posterior P(Know) for these skills: " + str(posterior_know) + "\nSTUDENT RESPONSE: " + str(student_response) + "\n" + "GAIN: " + str(gain) + "\nREWARD: " + str(reward) + "\n" + " EPSILON: " + str(agent.epsilon) + "\n"
    #             f.write(text)

    #         p_know = env.student_simulator.student_model.know[student_num].copy()

    #         if done:
    #             avg_p_know = np.mean(np.array(p_know))
    #             avg_p_knows.append(avg_p_know)

    #     print("episode: ", i, ", score: ", score, ", Avg_p_know: ", avg_p_know, ", TIMESTEPS: ", timesteps)
    #     scores.append(score)

    #     with open(CONSTANTS["ALGO"]+"_logs/rewards.txt", "a") as f:
    #         text = str(i) + "," + str(avg_p_know) + "\n"
    #         f.write(text)
    
    # final_p_know = np.array(env.student_simulator.student_model.know[student_num])
    # final_avg_p_know = np.mean(final_p_know)

    # print()
    # print(init_p_know)
    # print(final_p_know)
    # print("INITIAL AVERAGE P_KNOW: ", init_avg_p_know)
    # print("FINAL AVERAGE_P_KNOW: ", final_avg_p_know)
    # print("MIN AVERAGE_P_KNOW: ", min(avg_p_knows))
    # print("MAX AVERAGE_P_KNOW: ", max(avg_p_knows))
    # print("IMPROVEMENT: ", final_avg_p_know - init_avg_p_know)
    # print(final_p_know - init_p_know)
