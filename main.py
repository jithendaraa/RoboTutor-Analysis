import numpy as np
import torch
from helper import read_cta_table, get_kc_list_from_cta_table, init_kc_to_tutorID_dict, get_cta_tutor_ids, init_tutorID_to_kc_dict, get_skill_groups_info, get_proba, clear_files
from student_simulator import train_on_obs
from environment import StudentEnv
from DQN_agent import DQNAgent
from actor_critic_agent import ActorCriticAgent

STUDENT_ID = "new_student"
algo = "actor_critic"
START_EPISODE = 0
NUM_EPISODES = 1500
RUN = 0
AVG_OVER_EPISODES = 50
load = False
clear = True
MAX_TIMESTEPS = 250
learning_rate = 2e-5

cta_df = read_cta_table("Data/CTA.xlsx")
kc_list = get_kc_list_from_cta_table(cta_df)
u = len(kc_list)
num_skills = u
kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)
tutorID_to_kc_dict = init_tutorID_to_kc_dict(kc_to_tutorID_dict)

uniq_skill_groups, skill_group_to_activity_map = get_skill_groups_info(tutorID_to_kc_dict, kc_list)

print("DATA READING DONE...")

activity_bkt, activity_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(1.0, train_students=[STUDENT_ID])

student_id = student_id_to_number_map[STUDENT_ID]

initial_p_know = np.array(activity_bkt.know[student_id])
initial_avg_p_know = np.mean(initial_p_know)
initial_state = activity_bkt.know[student_id]
initial_state = np.array(initial_state)
state_size = initial_state.shape[0]
action_size = len(uniq_skill_groups)

env = StudentEnv(initial_state=initial_state, 
                activities=cta_tutor_ids, 
                activity_bkt=activity_bkt,
                tutorID_to_kc_dict=tutorID_to_kc_dict,
                student_id=student_id,
                skill_to_number_map=skill_to_number_map,
                skill_groups=uniq_skill_groups,
                skill_group_to_activity_map=skill_group_to_activity_map,
                action_size=action_size)

n_actions = len(uniq_skill_groups)

if algo == "dqn":
    agent = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=n_actions, input_dims=[state_size], lr=0.003, activity_to_skills_map=tutorID_to_kc_dict, kc_to_tutorID_dict=kc_to_tutorID_dict, cta_tutor_ids=cta_tutor_ids, kc_list=kc_list)

elif algo == "actor_critic":
    agent = ActorCriticAgent(alpha=learning_rate, 
                                input_dims=[state_size], 
                                activity_to_skills_map=tutorID_to_kc_dict, 
                                kc_to_tutorID_dict=kc_to_tutorID_dict, 
                                cta_tutor_ids=cta_tutor_ids, 
                                kc_list=kc_list,
                                skill_to_number_map=skill_to_number_map,
                                skill_groups=uniq_skill_groups,
                                skill_group_to_activity_map=skill_group_to_activity_map,
                                gamma=0.99,
                                layer1_size=4096, 
                                layer2_size=2048,
                                layer3_size=1024,
                                layer4_size=512,
                                n_actions=n_actions)

scores = []
eps_history = []

num_questions = 3
score = 0
p_know_dict = {}
p_know_per_question = [0] * num_skills

for i in range(NUM_EPISODES):
    p_know_for_question = []
    for j in range(num_questions):
        p_know_for_question.append(p_know_per_question)
    p_know_dict[i] = p_know_for_question

# empty contents. True -> empty else dont empty
clear_files(algo, clear)

PATH = "saved_model_parameters/" + algo + ".pth"
state = initial_state.copy()
avg_p_knows = []
avg_p_know = 0
avg_scores = []

if load == True:
    if algo == "dqn":
        agent.Q_eval.load_state_dict(torch.load(PATH))
        agent.Q_eval.eval()
    elif algo == "actor_critic":
        agent.actor_critic.load_state_dict(torch.load(PATH))
        agent.actor_critic.eval()


for i in range(START_EPISODE, NUM_EPISODES):

    if i % avg_over_episodes == 0 and i > 0:
        avg_score = np.mean(scores[max(0, i-avg_over_episodes): i+1])
        with open(algo+"_logs/avg_scores.txt", "a") as f:
            text = str(i/avg_over_episodes) + "," + str(avg_p_know) + "\n"
            f.write(text)
    
    if i > 0 and i % 500 == 0:
        if algo == "dqn":
            torch.save(agent.Q_eval.state_dict(), PATH)
            eps_history.append(agent.epsilon)
        elif algo == "actor_critic":
            torch.save(agent.actor_critic.state_dict(), PATH)

    score = 0
    state = env.reset()
    done = False
    timesteps = 0

    while done == False:
        timesteps += 1
        RUN += 1
        
        action, explore, sample_skill, activityName = agent.choose_action(state, explore=False)
        skillNums = uniq_skill_groups[action]

        prior_know = get_proba(action, activityName, tutorID_to_kc_dict, skill_to_number_map, env.activity_bkt.know[student_id])
        
        next_state, reward, student_response, done = env.step(action, timesteps, max_timesteps)

        posterior_know = get_proba(action, activityName, tutorID_to_kc_dict, skill_to_number_map, env.activity_bkt.know[student_id])
        
        if algo == "dqn":
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn(decrease_eps=True)

        elif algo == "actor_critic":
            agent.learn(state, reward, next_state, done)
        
        state = next_state
        score += reward
        gain = np.sum(np.array(posterior_know) - np.array(prior_know))/num_skills

        with open(algo+"_logs/run.txt", "a") as f:
            explore_status = "EXPLOIT"
            if explore:
                explore_status = "EXPLORE"
            text = "----------------------------------------------------------------------------------------------\n" + "RUN: " + str(RUN) + "\nTIMESTEPS: " + str(timesteps) + "\n EPISODE: " + str(i) + "\n" + explore_status + "\n" + "SAMPLED SKILL: " + str(sample_skill) + "\n" + "Action chosen: " + activityName + "\n" + " Skills: " + str(skillNums) + "\n" + " Prior P(Know) for these skills: " + str(prior_know) + "\n" + " Posterior P(Know) for these skills: " + str(posterior_know) + "\nSTUDENT RESPONSE: " + str(student_response) + "\n" + "GAIN: " + str(gain) + "\nREWARD: " + str(reward) + "\n" + " EPSILON: " + str(agent.epsilon) + "\n"
            f.write(text)

        p_know = env.activity_bkt.know[student_id].copy()
        p_know_dict[i][j] = p_know
        
        if done:
            avg_p_know = np.mean(np.array(p_know))
            avg_p_knows.append(avg_p_know)
        
            with open(algo+"_logs/avg_p_know.txt", "a") as f:
                text = str(i) + "," + str(avg_p_know) + "\n"
                f.write(text)

    print("episode: ", i, ", score: ", score, ", Avg_p_know: ", avg_p_know, ", TIMESTEPS: ", timesteps)
    scores.append(score)
    
    with open(algo+"_logs/rewards.txt", "a") as f:
        text = str(i) + "," + str(avg_p_know) + "\n"
        f.write(text)
    
    with open(algo+"_logs/timesteps.txt", "a") as f:
        text = str(i) + "," + str(timesteps) + "\n"
        f.write(text)
    
    with open(algo+"_logs/scores.txt", "a") as f:
        text = str(i) + "," + str(score) + "\n"
        f.write(text)

final_p_know = np.array(env.activity_bkt.know[student_id])
final_avg_p_know = np.mean(final_p_know)

print()
print(initial_p_know)
print(final_p_know)
print("INITIAL AVERAGE P_KNOW: ", initial_avg_p_know)
print("FINAL AVERAGE_P_KNOW: ", final_avg_p_know)
print("MIN AVERAGE_P_KNOW: ", min(avg_p_knows))
print("MAX AVERAGE_P_KNOW: ", max(avg_p_knows))
print("IMPROVEMENT: ", final_avg_p_know - initial_avg_p_know)
print(final_p_know - initial_p_know)
