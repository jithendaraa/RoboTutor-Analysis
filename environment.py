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
    def __init__(self, student_simulator, action_size, student_id='new_student', env_num=1, type=None, prints=True, area_rotation=None, CONSTANTS=None, matrix_area=None, matrix_posn=None):
        
        self.student_simulator = student_simulator
        self.student_id = student_id
        self.student_num = self.student_simulator.uniq_student_ids.index(self.student_id)
        self.action_size = action_size
        self.type = type
        self.area_rotation = area_rotation
        self.cycle_len = len(area_rotation.split('-'))
        self.CONSTANTS = CONSTANTS
        self.activity_num = None
        self.response = ""
        self.tutor_simulator = None
        self.attempts = 0

        self.set_initial_state(matrix_area=matrix_area, matrix_posn=matrix_posn)
        self.set_skill_groups()

        self.state_size = self.state.shape[0]
        self.observation_space = AbsSpace(self.state_size)
        self.action_space = Discrete(self.action_size)
        
        self.print_env_num(prints, env_num)
        
    def print_env_num(self, prints, env_num):
        if env_num is not None:
            if self.type != None and prints:
                print("Initialised RoboTutor environment number", env_num, "(Type " + str(self.type) + ") with", self.state_size, "states and", self.action_size, "actions")
            elif prints:
                print("Initialised RoboTutor environment number", env_num, "with", self.state_size, "states and", self.action_size, "actions")

    def set_skill_groups(self):
        if self.type == None:
            kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
            self.skill_groups                   = uniq_skill_groups
            self.skill_group_to_activity_map    = skill_group_to_activity_map
        
    def set_tutor_simulator(self, t1=None, t2=None, t3=None):
        if self.type == 1 or self.type == 2:
            if self.tutor_simulator == None:
                self.tutor_simulator = TutorSimulator(t1=t1, t2=t2, t3=t3, area_rotation=self.area_rotation, type=self.type, thresholds=True)
            else:
                self.tutor_simulator.set_thresholds(t1, t2, t3)
        elif self.type == 3:
            if self.tutor_simulator == None:
                self.tutor_simulator = TutorSimulator(area_rotation=self.area_rotation, type=self.type, thresholds=False)
        

    def set_initial_state(self, matrix_posn=None, matrix_area=None):
        
        student_model = self.student_simulator.student_model
        student_model_name = self.student_simulator.student_model_name

        if self.type == None or self.type == 1:
            if student_model_name == 'ItemBKT':
                pass
            
            elif student_model_name == 'ActivityBKT':
                self.state = np.array(student_model.know[self.student_num]).copy()
            
            elif student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
                self.state = np.array(student_model.alpha[self.student_num][-1]).copy()
        
        elif self.type == 2 or self.type == 3 or self.type == 4:

            if student_model_name == 'ItemBKT':
                pass
            
            elif student_model_name == 'ActivityBKT':
                pass

            elif student_model_name == 'hotDINA_skill':
                knows = student_model.alpha[self.student_num][-1]
                if matrix_posn != None: matrix_posn = [matrix_posn]
                else:   matrix_posn = [1]   # matrix position always starts with 1 unless mentioned otherwise; eg. a student was already placed at position 9
                if matrix_area == None: first_area = self.area_rotation[0]
                else:   # matrix area should be input as a number
                    area_rotation = self.area_rotation.split('-')
                    first_area = area_rotation[matrix_area % self.cycle_len]
                if first_area == 'L':
                    matrix_type = [1]  
                elif first_area == 'N':
                    matrix_type = [2]
                elif first_area == 'S':
                    matrix_type = [3]
                self.state = np.array(knows + matrix_type + matrix_posn)
                if self.type == 4:
                    self.state = np.array(knows + matrix_type)
            
            elif student_model_name == 'hotDINA_full':
                pass

    def reset(self):
        self.student_simulator.reset()
        self.state  = self.checkpoint_state.copy()
        return self.state.copy()

    def checkpoint(self):
        self.student_simulator.checkpoint()
        student_model = self.student_simulator.student_model    #  Saves some values as checkpoints so env.reset() resets env values to checkpoint states
        student_model_name = self.student_simulator.student_model_name
        self.checkpoint_state = self.state.copy()
    
    # TODO
    def item_bkt_step(self):
        if self.type == None:
            pass
        elif self.type == 1:
            pass
        elif self.type == 2:
            pass
        elif self.type == 3:
            pass
        elif self.type == 4:
            pass
        elif self.type == 5:
            pass
    
    # TODO
    def activity_bkt_step(self, activity=None):
        # 1. Get student prior knowledge (before attempting the question)
        # 2. Simulate student with current model params and get student response (predictions) according to full responsibility
        # 3. Update model params based on simulated student response and get the posterior know
        if self.type == None:
            prior_know = self.student_simulator.student_model.know[self.student_num].copy()
            correct_preds, min_correct_preds = self.student_simulator.student_model.predict_percent_corrects(student_ids, skills)
            student_response = correct_preds    # should be min_correct_preds for "hardest_skill" responsibility
            self.student_simulator.student_model.update(correct_preds, [self.student_num], [activity])
            posterior_know = self.student_simulator.student_model.know[self.student_num].copy()
            next_state = np.array(posterior_know.copy())
            self.state = next_state.copy()

        elif self.type == 1:
            pass
        elif self.type == 2:
            pass
        elif self.type == 3:
            pass
        elif self.type == 4:
            pass
        elif self.type == 5:
            pass

        return next_state, prior_know, student_response, posterior_know

    def hotDINA_skill_step(self, action, prints, activity_num=None):

        done = False
        p_know_activity = None
        village = self.student_simulator.village
        observations = self.student_simulator.observations
        student_model_name = self.student_simulator.student_model_name
        new_student_params = self.student_simulator.new_student_params
        
        if self.type == 1 or self.type == 2:
            t1, t2, t3 = action[0], action[1], action[2]
            self.set_tutor_simulator(t1=t1, t2=t2, t3=t3)
        else:
            self.set_tutor_simulator()

        if self.type == None:
            prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            student_response = self.student_simulator.student_model.predict_response(activity_num, self.student_num, update=True)
            posterior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            next_state = np.array(posterior_know.copy())
            self.state = next_state.copy()
        
        elif self.type == 1:
            avg_performance_given_thresholds = []
            avg_posterior_know = None
            avg_prior_know = None
            posterior_know = []
            prior_know = None
            for _ in range(5):
                student_simulator = StudentSimulator(village, observations, student_model_name, new_student_params, prints=False)
                tutor_simulator = TutorSimulator(t1=t1, t2=t2, t3=t3, area_rotation=self.area_rotation, type=self.type, thresholds=True)
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
        
        elif self.type == 2:
            self.tutor_simulator.set_thresholds(t1, t2, t3)
            if self.activity_num != None:   p_know_activity = self.student_simulator.student_model.get_p_know_activity(self.student_num, self.activity_num)
            x, y, area, activity_name = self.tutor_simulator.get_next_activity(p_know_activity=p_know_activity, prev_activity_num=self.activity_num, response=str(self.response), prints=False)
            self.activity_num = self.student_simulator.uniq_activities.index(activity_name)
            prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            student_response = self.student_simulator.student_model.predict_response(self.activity_num, self.student_num, update=True)
            posterior_know = np.array(self.student_simulator.student_model.alpha[self.student_num][-1])
            posterior_avg_know = np.mean(posterior_know)
            next_matrix_type = self.tutor_simulator.get_matrix_area()
            p_know_activity = self.student_simulator.student_model.get_p_know_activity(self.student_num, self.activity_num)
            next_matrix_posn = self.tutor_simulator.get_matrix_posn(p_know_act=p_know_activity)
            next_state = np.array(posterior_know.tolist() + [next_matrix_type] +[next_matrix_posn])
        
        elif self.type == 3:
            if action == 0: decision = "prev"
            elif action == 1: decision = "same"
            elif action == 2: decision = 'next'
            elif action == 3: decision = 'next_next'

            x, y, area, activity_name = self.tutor_simulator.get_next_activity(decision=decision, prev_activity_num=self.activity_num, prints=False)
            self.activity_num = self.student_simulator.uniq_activities.index(activity_name)
            prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            student_response = self.student_simulator.student_model.predict_response(self.activity_num, self.student_num, update=True)
            posterior_know = np.array(self.student_simulator.student_model.alpha[self.student_num][-1])
            posterior_avg_know = np.mean(posterior_know)
            next_matrix_type = self.tutor_simulator.get_matrix_area()
            p_know_activity = self.student_simulator.student_model.get_p_know_activity(self.student_num, self.activity_num)
            next_matrix_posn = self.tutor_simulator.get_matrix_posn(p_know_act=p_know_activity)
            next_state = np.array(posterior_know.tolist() + [next_matrix_type] +[next_matrix_posn])

        elif self.type == 4:
            # Action is the index into uniq_activities
            activity_num = action
            prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            student_response = self.student_simulator.student_model.predict_response(activity_num, self.student_num, update=True)
            posterior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            self.attempts += 1
            area_rotation = self.area_rotation.split('-')
            matrix_type = area_rotation[self.attempts % self.cycle_len]
            activity_name = self.student_simulator.uniq_activities[activity_num]
            if matrix_type == 'L':
                matrix_num = [1]
            elif matrix_type == 'N':
                matrix_num = [2]
            elif matrix_type == 'S':
                matrix_num = [3]
            next_state = np.array(posterior_know + matrix_num)
            self.state = next_state.copy()

        elif self.type == 5:
            pass
            
        return next_state, student_response, done, prior_know, posterior_know
    
    def hotDINA_full_step(self, activity_num):

        done = False
        
        if self.type == None:
            prior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            _, predicted_response = self.student_simulator.student_model.predict_response(activity_num, self.student_num, update=True)
            posterior_know = self.student_simulator.student_model.alpha[self.student_num][-1].copy()
            student_response = predicted_response
            next_state = np.array(posterior_know.copy())
            self.state = next_state.copy()
        
        elif self.type == 1:
            pass
        elif self.type == 2:
            pass
        elif self.type == 3:
            pass
        elif self.type == 4:
            pass
        elif self.type == 5:
            pass
        
        return next_state, student_response, done, prior_know, posterior_know
        
    def step(self, action, max_timesteps, timesteps=None, activityName=None, bayesian_update=True, plot=False, prints=False):

        done = False
        student_ids = [self.student_id]
        student_nums = [self.student_num]
        student_model_name = self.student_simulator.student_model_name
        activity_num = None

        if self.type == None:
            skill_group = self.skill_groups[action]
            skills = [skill_group]
            activity = self.student_simulator.underscore_to_colon_tutor_id_dict[activityName]
            activity_num = self.student_simulator.uniq_activities.index(activity)
            activity_nums = [activity_num] 

        if student_model_name == 'ItemBKT':
            self.item_bkt_step()
        
        elif student_model_name == 'ActivityBKT':
            next_state, prior_know, student_response, posterior_know = self.activity_bkt_step(activity=activity)
        
        elif student_model_name == 'hotDINA_skill':
            next_state, student_response, done, prior_know, posterior_know = self.hotDINA_skill_step(action, prints, activity_num=activity_num)
            
        elif self.student_simulator.student_model_name == 'hotDINA_full':
            self.hotDINA_full_step(activity_num)
        
        if timesteps != None and timesteps >= max_timesteps:
            done = True

        avg_prior_know = np.mean(np.array(prior_know))
        avg_posterior_know = np.mean(np.array(posterior_know))
        reward = 1000 * (avg_posterior_know - avg_prior_know) 

        return next_state, reward, student_response, done, posterior_know