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
    def __init__(self, student_simulator, skill_groups, skill_group_to_activity_map, action_size, student_id='new_student', env_num=1):
        
        self.student_simulator = student_simulator
        self.student_id = student_id
        self.student_num = self.student_simulator.uniq_student_ids.index(self.student_id)
        print(self.student_id, self.student_num)
        self.skill_groups                   = skill_groups
        self.skill_group_to_activity_map    = skill_group_to_activity_map
        
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
            self.initial_state = np.array(student_model.know[student_num]).copy()
    
        elif self.student_simulator.student_model_name == 'hotDINA_skill':
            self.initial_state = np.array(student_model.knews[student_num]).copy()

        self.state = self.initial_state.copy()


    def reset(self):
        student_model = self.student_simulator.student_model
        self.state = self.checkpoint_know.copy()
        if self.student_simulator.student_model_name == 'ActivityBKT':
            self.student_simulator.student_model.know[self.student_num] = self.checkpoint_know.tolist().copy()
            self.student_simulator.student_model.know_act[self.student_num] = self.checkpoint_know_act.tolist().copy()
            print("RESSET CHECKPOINT LP: ", len(self.checkpoint_learning_progress[self.student_id].copy()))
            self.student_simulator.student_model.learning_progress = self.checkpoint_learning_progress.copy()
        
        return self.state.copy()

    def checkpoint(self):
        #  Saves some values so env.reset() resets env values to checkpoint values
        self.checkpoint_know = self.student_simulator.student_model.know[self.student_num].copy()
        self.checkpoint_know_act = self.student_simulator.student_model.know_act[self.student_num].copy()
        self.checkpoint_learning_progress = self.student_simulator.student_model.learning_progress.copy()
        print("CHECKPOINT KNOW:", self.checkpoint_know)
    
    def step(self, action, timesteps, max_timesteps, activityName):
        """
            action: int, activities[action] refers to the activity that the RL policy suggests/outputs.
        """
        done = False
        student_ids = [self.student_id]
        student_nums = [self.student_num]
        skill_group = self.skill_groups[action]
        skills = [skill_group]
        activity = self.student_simulator.underscore_to_colon_tutor_id_dict[activityName]

        if self.student_simulator.student_model_name == 'ActivityBKT':
        
            # Simulate student with current BKT params and get student response (predictions) according to full responsibility and blame-worst responsibility
            correct_preds, min_correct_preds = self.student_simulator.student_model.predict_percent_correct(student_ids, skills)

            # should be min_correct_preds for "blame worst" responsibility
            student_response = correct_preds 

            # P(Know) as a list before attempting the question (before doing BKT update based on student_response)
            prior_know = self.student_simulator.student_model.know[self.student_num].copy()

            # BKT update based on student_response aka %correct
            self.student_simulator.student_model.update(activity_observations=student_response, 
                                                        student_nums=student_nums,  
                                                        activities=[activity])

            # Get next state as updated P(Know) based on student_response. Set next_state as current_state
            posterior_know = self.student_simulator.student_model.know[self.student_num].copy()
            next_state = self.student_simulator.student_model.know[self.student_num].copy()
            next_state = np.array(next_state)
            self.state = next_state.copy()

        # Get avg P(Know) before and after student attempts the activities
        avg_prior_know = np.mean(np.array(prior_know))
        avg_posterior_know = np.mean(np.array(posterior_know))

        # Reward after each attempt/opportunity
        reward = 1000 * (avg_posterior_know - avg_prior_know) # reward = 100 * np.mean(np.array(self.activity_bkt.know))

        if timesteps >= max_timesteps:
            done = True

        # print('next_state @env.py: ', next_state)
        return next_state, reward, student_response[0], done, posterior_know