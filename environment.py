import numpy as np

from student_simulator import StudentSimulator
from tutor_simulator import TutorSimulator

from reader import *
from helper import *
# The Environment that the RL agent interacts with
# Given an action returns the new_state, reward, student_response, done and posterior know (P(Know) after student attempt)

class AbsSpace():
    def __init__(self, size):
        self.shape = (size, )
        
class Discrete():
    def __init__(self, size):
        self.shape = (size, )

class StudentEnv():
    def __init__(self, student_simulator, action_size, student_id='new_student', env_num=1, type=None, prints=True, area_rotation=None, CONSTANTS=None):
        
        self.student_simulator = student_simulator
        self.student_id = student_id
        self.student_num = self.student_simulator.uniq_student_ids.index(self.student_id)
        self.action_size = action_size
        self.set_initial_state()
        self.state_size = self.state.shape[0]
        self.observation_space = AbsSpace(self.state_size)
        self.action_space = Discrete(self.action_size)
        self.type = type
        self.area_rotation = area_rotation
        self.CONSTANTS = CONSTANTS
        
        if self.type == None:
            kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
            self.skill_groups                   = skill_groups
            self.skill_group_to_activity_map    = skill_group_to_activity_map

        if env_num is not None:
            if type != None and prints:
                print("Initialised RoboTutor environment number", env_num, "(Type " + str(type) + ") with", self.state_size, "states and", self.action_size, "actions")
            elif prints:
                print("Initialised RoboTutor environment number", env_num, "with", self.state_size, "states and", self.action_size, "actions")
    
    def set_initial_state(self):
        student_model = self.student_simulator.student_model
        student_model_name = self.student_simulator.student_model_name
        if student_model_name == 'ActivityBKT':
            self.state = np.array(student_model.know[self.student_num]).copy()
        elif student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
            self.state = np.array(student_model.alpha[self.student_num][-1]).copy()

    def reset(self):
        self.student_simulator.reset()
        self.state  = self.checkpoint_state.copy()
        return self.state.copy()

    def checkpoint(self):
        self.student_simulator.checkpoint()
        student_model = self.student_simulator.student_model    #  Saves some values as checkpoints so env.reset() resets env values to checkpoint states
        student_model_name = self.student_simulator.student_model_name

        if student_model_name == 'ActivityBKT':
            self.checkpoint_state = student_model.know[self.student_num].copy()
       
        elif student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
            self.checkpoint_state = student_model.alpha[self.student_num][-1].copy()

    def step(self, action, max_timesteps, timesteps=None, activityName=None, bayesian_update=True, plot=False, prints=False):
        """
            action: int, activities[action] refers to the activity that the RL policy suggests/outputs.
        """
        done = False
        student_ids = [self.student_id]
        student_nums = [self.student_num]
        if self.type == None:
            skill_group = self.skill_groups[action]
            skills = [skill_group]
            activity = self.student_simulator.underscore_to_colon_tutor_id_dict[activityName]
            activity_num = self.student_simulator.uniq_activities.index(activity)
            activity_nums = [activity_num] 

        # 1. Get student prior knowledge (before attempting the question)
        # 2. Simulate student with current model params and get student response (predictions) according to full responsibility
        # 3. Update model params based on simulated student response and get the posterior know
        if self.student_simulator.student_model_name == 'ActivityBKT':
            prior_know = self.student_simulator.student_model.know[self.student_num].copy()
            correct_preds, min_correct_preds = self.student_simulator.student_model.predict_percent_corrects(student_ids, skills)
            student_response = correct_preds    # should be min_correct_preds for "hardest_skill" responsibility
            self.student_simulator.student_model.update(correct_preds, student_nums,[activity])
            posterior_know = self.student_simulator.student_model.know[self.student_num].copy()
            next_state = np.array(posterior_know.copy())
            self.state = next_state.copy()
        
        elif self.student_simulator.student_model_name == 'hotDINA_skill':
            if self.type == None:
                prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
                student_response = self.student_simulator.student_model.predict_response(activity_num, self.student_num, update=True)
                posterior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
                next_state = np.array(posterior_know.copy())
                self.state = next_state.copy()
            
            elif self.type == 1:
                t1 = action[0]
                t2 = action[1]
                t3 = action[2]
                reward = 0.0
                
                village = self.student_simulator.village
                observations = self.student_simulator.observations
                student_model_name = self.student_simulator.student_model_name
                new_student_params = self.student_simulator.new_student_params

                avg_performance_given_thresholds = []
                avg_posterior_know = None
                avg_prior_know = None
                posterior_know = []
                prior_know = None

                for _ in range(5):
                    student_simulator = StudentSimulator(village, observations, student_model_name, new_student_params, prints=False)
                    tutor_simulator = TutorSimulator(t1, t2, t3, area_rotation=self.area_rotation, type=self.type, thresholds=True)
                    
                    prior_know = np.mean(student_simulator.student_model.alpha[self.student_num][-1].copy())
                    performance_given_thresholds = evaluate_performance_thresholds(student_simulator, tutor_simulator, prints=prints, CONSTANTS=self.CONSTANTS)
                    posterior_know.append(student_simulator.student_model.alpha[self.student_num][-1].copy())

                    avg_performance_given_thresholds.append(performance_given_thresholds)

                avg_performance_given_thresholds = np.mean(avg_performance_given_thresholds, axis=0)
                posterior_know = np.mean(posterior_know, axis=0)
                avg_posterior_know = np.mean(posterior_know)
                self.reset()
                next_state = self.state.copy()
                student_response = None
                done = True
        
        elif self.student_simulator.student_model_name == 'hotDINA_full':
            prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            _, predicted_response = self.student_simulator.student_model.predict_response(activity_num, self.student_num, update=True)
            posterior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            student_response = predicted_response
            next_state = np.array(posterior_know.copy())
            self.state = next_state.copy()
        
        if timesteps != None and timesteps >= max_timesteps:
            done = True

        avg_prior_know = np.mean(np.array(prior_know))
        avg_posterior_know = np.mean(np.array(posterior_know))
        reward = 1000 * (avg_posterior_know - avg_prior_know) 

        return next_state, reward, student_response, done, posterior_know