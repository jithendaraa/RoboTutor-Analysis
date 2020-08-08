import numpy as np
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
                "MAX_TIMESTEPS"         : 150,
                "LEARNING_RATE"         : 2e-5,
                "STATE_SIZE"            : 18,
                "ACTION_SIZE"           : 33,
                "GAMMA"                 : 0.99,
                "NUM_OBS"               : "all"
            }


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--observations", help="NUM_ENTRIES that have to be extracted from a given transactions table. Should be a number or 'all'. If inputted number > total records for the village, this will assume a value of 'all'", required=False)
parser.add_argument("-v", "--village_num", help="village_num whose transactions data has to be extracted, should be between 114 and 141", required=False)
args = parser.parse_args()

if args.village_num != None:
    CONSTANTS["VILLAGE"] = args.village_num
if args.observations != None:
    CONSTANTS["NUM_OBS"] = args.observations

clear_files(CONSTANTS["ALGO"], CONSTANTS["CLEAR_FILES"])

# Get some important data 
kc_list, num_skills, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map = read_data()

student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"])
# activity_bkt, activity_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(1.0, train_students=[CONSTANTS["STUDENT_ID"]])

# student_id = student_id_to_number_map[CONSTANTS["STUDENT_ID"]]
# initial_state = np.array(activity_bkt.know[student_id])

# env = StudentEnv(initial_state=initial_state, 
#                 activities=cta_tutor_ids, 
#                 activity_bkt=activity_bkt,
#                 tutorID_to_kc_dict=tutorID_to_kc_dict,
#                 student_id=student_id,
#                 skill_to_number_map=skill_to_number_map,
#                 skill_groups=uniq_skill_groups,
#                 skill_group_to_activity_map=skill_group_to_activity_map,
#                 action_size=CONSTANTS["ACTION_SIZE"])

# init_p_know = np.array(activity_bkt.know[student_id])
# init_avg_p_know = np.mean(init_p_know)

# if CONSTANTS["ALGO"] == "dqn":
#     agent = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=CONSTANTS["ACTION_SIZE"], input_dims=[CONSTANTS["STATE_SIZE"]], lr=0.003, activity_to_skills_map=tutorID_to_kc_dict, kc_to_tutorID_dict=kc_to_tutorID_dict, cta_tutor_ids=cta_tutor_ids, kc_list=kc_list)

# elif CONSTANTS["ALGO"] == "actor_critic":
#     agent = ActorCriticAgent(alpha=CONSTANTS["LEARNING_RATE"], 
#                                 input_dims=[CONSTANTS["STATE_SIZE"]], 
#                                 activity_to_skills_map=tutorID_to_kc_dict, 
#                                 kc_to_tutorID_dict=kc_to_tutorID_dict, 
#                                 cta_tutor_ids=cta_tutor_ids, 
#                                 kc_list=kc_list,
#                                 skill_to_number_map=skill_to_number_map,
#                                 skill_groups=uniq_skill_groups,
#                                 skill_group_to_activity_map=skill_group_to_activity_map,
#                                 gamma=CONSTANTS["GAMMA"],
#                                 layer1_size=4096, 
#                                 layer2_size=2048,
#                                 layer3_size=1024,
#                                 layer4_size=512,
#                                 n_actions=CONSTANTS["ACTION_SIZE"])

# scores = []
# # can be used to see how epsilon changes with each timestep/opportunity
# eps_history = []

# num_questions = 3
# score = 0

# PATH = "saved_model_parameters/" + CONSTANTS["ALGO"] + ".pth"
# avg_p_knows = []
# avg_p_know = 0
# avg_scores = []

# if CONSTANTS["LOAD"] == True:
#     if CONSTANTS["ALGO"] == "dqn":
#         agent.Q_eval.load_state_dict(torch.load(PATH))
#         agent.Q_eval.eval()
#     elif CONSTANTS["ALGO"] == "actor_critic":
#         agent.actor_critic.load_state_dict(torch.load(PATH))
#         agent.actor_critic.eval()


# for i in range(CONSTANTS["START_EPISODE"], CONSTANTS["NUM_EPISODES"]):

#     if i % CONSTANTS["AVG_OVER_EPISODES"] == 0 and i > 0:
#         avg_score = np.mean(scores[max(0, i-CONSTANTS["AVG_OVER_EPISODES"]): i+1])
#         with open(CONSTANTS["ALGO"]+"_logs/avg_scores.txt", "a") as f:
#             text = str(i/CONSTANTS["AVG_OVER_EPISODES"]) + "," + str(avg_p_know) + "\n"
#             f.write(text)
    
#     if i > 0 and i % 500 == 0:
#         if CONSTANTS["ALGO"] == "dqn":
#             torch.save(agent.Q_eval.state_dict(), PATH)
#             eps_history.append(agent.epsilon)
#         elif CONSTANTS["ALGO"] == "actor_critic":
#             torch.save(agent.actor_critic.state_dict(), PATH)

#     score = 0
#     state = env.reset()
#     done = False
#     timesteps = 0

#     while done != True:
#         timesteps += 1
#         CONSTANTS["RUN"] += 1
        
#         action, explore, sample_skill, activityName = agent.choose_action(state, explore=False)
#         skillNums = uniq_skill_groups[action]

#         prior_know = get_proba(action, activityName, tutorID_to_kc_dict, skill_to_number_map, env.activity_bkt.know[student_id])
#         next_state, reward, student_response, done, posterior = env.step(action, timesteps, CONSTANTS["MAX_TIMESTEPS"])
#         posterior_know = get_proba(action, activityName, tutorID_to_kc_dict, skill_to_number_map, env.activity_bkt.know[student_id])

#         if CONSTANTS["ALGO"] == "dqn":
#             agent.store_transition(state, action, reward, next_state, done)
#             agent.learn(decrease_eps=True)

#         elif CONSTANTS["ALGO"] == "actor_critic":
#             agent.learn(state, reward, next_state, done)
        
#         state = next_state
#         score += reward
#         gain = np.sum(np.array(posterior_know) - np.array(prior_know))/num_skills

#         with open(CONSTANTS["ALGO"]+"_logs/run.txt", "a") as f:
#             explore_status = "EXPLOIT"
#             if explore:
#                 explore_status = "EXPLORE"
#             text = "----------------------------------------------------------------------------------------------\n" + "RUN: " + str(CONSTANTS["RUN"]) + "\nTIMESTEPS: " + str(timesteps) + "\n EPISODE: " + str(i) + "\n" + explore_status + "\n" + "SAMPLED SKILL: " + str(sample_skill) + "\n" + "Action chosen: " + activityName + "\n" + " Skills: " + str(skillNums) + "\n" + " Prior P(Know) for these skills: " + str(prior_know) + "\n" + " Posterior P(Know) for these skills: " + str(posterior_know) + "\nSTUDENT RESPONSE: " + str(student_response) + "\n" + "GAIN: " + str(gain) + "\nREWARD: " + str(reward) + "\n" + " EPSILON: " + str(agent.epsilon) + "\n"
#             f.write(text)

#         p_know = env.activity_bkt.know[student_id].copy()
        
#         if done:
#             avg_p_know = np.mean(np.array(p_know))
#             avg_p_knows.append(avg_p_know)

#     print("episode: ", i, ", score: ", score, ", Avg_p_know: ", avg_p_know, ", TIMESTEPS: ", timesteps)
#     scores.append(score)
    
#     with open(CONSTANTS["ALGO"]+"_logs/rewards.txt", "a") as f:
#         text = str(i) + "," + str(avg_p_know) + "\n"
#         f.write(text)

# final_p_know = np.array(env.activity_bkt.know[student_id])
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
