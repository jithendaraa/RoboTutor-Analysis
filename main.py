import numpy as np
import pandas as pd
import torch
import argparse
import pickle

from helper import *
from reader import *

# RL Agents
from RL_agents.DQN_agent import DQNAgent
from RL_agents.actor_critic_agent import ActorCriticAgent

# Environment and Student Simulator
from environment import StudentEnv
from student_simulator import StudentSimulator

CONSTANTS = {
                "LOAD"                  : False,
                "CLEAR_FILES"           : False,
                "STUDENT_ID"            : "new_student",
                "ALGO"                  : "actor_critic",
                "STUDENT_MODEL_NAME"    : "ActivityBKT",
                "VILLAGE"               : "130",
                "START_EPISODE"         : 0,
                "NUM_EPISODES"          : 1000,
                "RUN"                   : 0,
                "AVG_OVER_EPISODES"     : 50,
                "MAX_TIMESTEPS"         : 300,
                "LEARNING_RATE"         : 2e-5,
                "STATE_SIZE"            : 22,
                "ACTION_SIZE"           : 43,
                "ACTION_TYPE"           : 'per_skill_group',
                "GAMMA"                 : 0.99,
                "NUM_OBS"               : "all",
                'NUM_SKILL_GROUPS'      : 43,
                'EPSILON'               : 1.0
            }

def set_constants(args):
    CONSTANTS["NUM_OBS"]            = args.observations
    CONSTANTS["VILLAGE"]            = args.village_num
    CONSTANTS['STUDENT_ID']         = args.student_id
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['ACTION_TYPE']        = args.action_type

    cta_df = read_cta_table("Data/CTA.xlsx")
    kc_list = get_kc_list_from_cta_table(cta_df)
    num_skills = len(kc_list)

    Q = pd.read_csv('../hotDINA/qmatrix.txt', header=None).to_numpy()
    skill_groups = []
    for row in Q:
        if len(skill_groups) == 0 or row[:num_skills].tolist() not in skill_groups:
            skill_groups.append(row[:num_skills].tolist())
    
    num_skill_groups = len(skill_groups)
    CONSTANTS['NUM_SKILL_GROUPS'] = num_skill_groups

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
        agent = agent = ActorCriticAgent(alpha=CONSTANTS["LEARNING_RATE"], 
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

if __name__ == '__main__':

    clear_files(CONSTANTS["ALGO"], CONSTANTS["CLEAR_FILES"])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", help="NUM_ENTRIES that have to be extracted from a given transactions table. Should be a number or 'all'. If inputted number > total records for the village, this will assume a value of 'all'", default='1000')
    parser.add_argument("-v", "--village_num", help="village_num whose transactions data has to be extracted, should be between 114 and 141", default="130")
    parser.add_argument('-m', '--matrix_type', help="math literacy or stories", default='math')
    parser.add_argument('-sid', '--student_id', help="Student id", required=False, default='new_student')
    parser.add_argument('-smn', '--student_model_name', help="Name of student model: (ItemBKT, ActivityBKT, hotDINA_skill)", required=False, default='ActivityBKT')
    parser.add_argument('-at', '--action_type', help="per_skill_group (43), per_activity (1712), transition (4; prev, same, next, next-next), thresholds, transition-threshold. All are single actions except 'transition threshold which takes 2 actions from a (num_thresholds,num_activities) actions space' ", default='per_skill_group')
    args = parser.parse_args()

    set_constants(args)
    matrix_type = args.matrix_type
    student_id = args.student_id

    student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"])

    kc_list = student_simulator.kc_list
    cta_df = student_simulator.cta_df
    kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
    cta_tutor_ids = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)
    tutorID_to_kc_dict = get_tutorID_to_kc_dict(kc_to_tutorID_dict)
    uniq_skill_groups, skill_group_to_activity_map = get_skill_groups_info(tutorID_to_kc_dict, kc_list)
    student_num = student_simulator.uniq_student_ids.index(student_id)
    num_skills = len(kc_list)

    env = StudentEnv(student_simulator=student_simulator,
                    skill_groups=uniq_skill_groups,
                    skill_group_to_activity_map = skill_group_to_activity_map,
                    action_size=CONSTANTS["ACTION_SIZE"],
                    student_id=student_id)

    init_p_know = get_init_know(student_simulator, student_id)
    init_avg_p_know = np.mean(init_p_know)
    set_state_size(num_skills)

    agent = init_agent(uniq_skill_groups, student_simulator, skill_group_to_activity_map, tutorID_to_kc_dict, kc_to_tutorID_dict, cta_tutor_ids, kc_list)

    scores = []
    eps_history = []    # can be used to see how epsilon changes with each timestep/opportunity
    score = 0
    avg_p_knows = []
    avg_p_know = 0
    avg_scores = []

    PATH = "saved_model_parameters/" + CONSTANTS["ALGO"] + ".pth"
    agent = load_params(PATH, agent)

    for i in range(10):

        score = 0
        state = env.reset()
        done = False
        timesteps = 0

        output_avg_p_know(i, CONSTANTS["AVG_OVER_EPISODES"], scores, CONSTANTS["ALGO"]+"_logs/avg_scores.txt", avg_p_know)

        if i > 0 and i % 500 == 0:
            save_params(PATH, agent)

        while done != True:
            timesteps += 1
            CONSTANTS["RUN"] += 1

            action, explore, sample_skill, activityName = agent.choose_action(state, explore=False)
            skillNums = uniq_skill_groups[action]

            prior_know = skill_probas(activityName, tutorID_to_kc_dict, kc_list, env.student_simulator.student_model.know[student_num])
            next_state, reward, student_response, done, posterior = env.step(action, timesteps, CONSTANTS["MAX_TIMESTEPS"], activityName)
            posterior_know = skill_probas(activityName, tutorID_to_kc_dict, kc_list, env.student_simulator.student_model.know[student_num])

            if CONSTANTS["ALGO"] == "dqn":
                agent.store_transition(state, action, reward, next_state, done)
                agent.learn(decrease_eps=True)

            elif CONSTANTS["ALGO"] == "actor_critic":
                agent.learn(state, reward, next_state, done)

            state = next_state
            score += reward
            gain = np.sum(np.array(posterior_know) - np.array(prior_know))/num_skills

            # with open(CONSTANTS["ALGO"]+"_logs/run.txt", "a") as f:
            #     explore_status = "EXPLOIT"
            #     if explore:
            #         explore_status = "EXPLORE"
            #     text = "----------------------------------------------------------------------------------------------\n" + "RUN: " + str(CONSTANTS["RUN"]) + "\nTIMESTEPS: " + str(timesteps) + "\n EPISODE: " + str(i) + "\n" + explore_status + "\n" + "SAMPLED SKILL: " + str(sample_skill) + "\n" + "Action chosen: " + activityName + "\n" + " Skills: " + str(skillNums) + "\n" + " Prior P(Know) for these skills: " + str(prior_know) + "\n" + " Posterior P(Know) for these skills: " + str(posterior_know) + "\nSTUDENT RESPONSE: " + str(student_response) + "\n" + "GAIN: " + str(gain) + "\nREWARD: " + str(reward) + "\n" + " EPSILON: " + str(agent.epsilon) + "\n"
            #     f.write(text)
            p_know = env.student_simulator.student_model.know[student_num].copy()

            if done:
                avg_p_know = np.mean(np.array(p_know))
                avg_p_knows.append(avg_p_know)

        print("episode: ", i, ", score: ", score, ", Avg_p_know: ", avg_p_know, ", TIMESTEPS: ", timesteps)
        scores.append(score)

    #     with open(CONSTANTS["ALGO"]+"_logs/rewards.txt", "a") as f:
    #         text = str(i) + "," + str(avg_p_know) + "\n"
    #         f.write(text)
    
    final_p_know = np.array(env.student_simulator.student_model.know[student_id])
    final_avg_p_know = np.mean(final_p_know)

    print()
    print(init_p_know)
    print(final_p_know)
    print("INITIAL AVERAGE P_KNOW: ", init_avg_p_know)
    print("FINAL AVERAGE_P_KNOW: ", final_avg_p_know)
    print("MIN AVERAGE_P_KNOW: ", min(avg_p_knows))
    print("MAX AVERAGE_P_KNOW: ", max(avg_p_knows))
    print("IMPROVEMENT: ", final_avg_p_know - init_avg_p_know)
    print(final_p_know - init_p_know)
