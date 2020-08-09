import numpy as np
from reader import *

# The Environment that the RL agent interacts with
# Given an action returns the new_state, reward, student_response, done and posterior know (P(Know) after student attempt)

class AbsSpace():
    def __init__(self, size):
        self.shape = (size, )
        
class Discrete():
    def __init__(self, size):
        self.shape = (size, )

class StudentEnv():
    def __init__(self, student_simulator, action_size, student_id='new_student', env_num=1):
        
        self.student_simulator = student_simulator
        self.student_id = student_id
        # can be taken from unique rows of self.student_simulator.Q
        # self.skill_groups                   = skill_groups
        kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, self.skill_group_to_activity_map = read_data()
        # self.skill_group_to_activity_map    = skill_group_to_activity_map
        
        self.set_initial_state()
        self.action_size                    = action_size
        self.state_size                     = self.state.shape[0]
        self.observation_space              = AbsSpace(self.state_size)
        self.action_space                   = Discrete(self.action_size)

        if env_num is not None:
            print("Initialised RoboTutor environment number", env_num, "with", self.state_size, "states and", self.action_size, "actions")
    
    def set_initial_state(self):

        student_model = self.student_simulator.student_model
        student_num = self.student_simulator.uniq_student_ids.index(self.student_id)
        
        if self.student_simulator.student_model_name == 'ActivityBKT':
            self.initial_state = np.array(student_model.know[student_num])
    
        elif self.student_simulator.student_model_name == 'hotDINA_skill':
            self.initial_state = np.array(student_model.knews[student_num])

        self.state = self.initial_state

    # def reset(self):
    #     self.activity_bkt.know[self.student_id] = self.initial_state.tolist().copy()
    #     return self.initial_state
    
    # def step(self, action, timesteps, max_timesteps):
    #     """
    #         action: int, activities[action] refers to the activity that the RL policy suggests/outputs.
    #     """
    #     done = False
    #     student_ids = [self.student_id]
    #     skills = []
    #     skill_group = self.skill_groups[action]
    #     skills = [skill_group]
    #     # Simulate student with current BKT params and get student response (predictions) according to full responsibility and blame-worst responsibility
    #     correct_preds, min_correct_preds = self.activity_bkt.predict_percent_correct(student_ids, skills)
        
    #     # should be min_correct_preds for "blame worst" responsibility
    #     student_response = correct_preds 
        
    #     # P(Know) as a list before attempting the question (before doing BKT update based on student_response)
    #     prior_know = self.activity_bkt.know[self.student_id].copy()

    #     # BKT update based on student_response aka %correct
    #     self.activity_bkt.update(activity_observations=student_response, student_ids=student_ids, skills=skills)

    #     # Get next state as updated P(Know) based on student_response. Set next_state as current_state
    #     next_state = self.activity_bkt.know[self.student_id]
    #     next_state = np.array(next_state)
    #     self.state = next_state.copy()

    #     # Get posterior P(Know) ie P(Know) after BKT update from student_response 
    #     posterior_know = self.activity_bkt.know[self.student_id].copy()

    #     # Get avg P(Know) before and after student attempts the activities
    #     avg_prior_know = np.mean(np.array(prior_know))
    #     avg_posterior_know = np.mean(np.array(posterior_know))

    #     # Reward after each attempt/opportunity
    #     reward = 1000 * (avg_posterior_know - avg_prior_know) # reward = 100 * np.mean(np.array(self.activity_bkt.know))

    #     if reward <= 0.0:
    #         done = True

    #     # if timesteps >= max_timesteps:
    #     #     done = True

    #     return next_state, reward, student_response[0], done, posterior_know