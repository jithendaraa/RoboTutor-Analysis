import sys
sys.path.append('..')
import pandas as pd

from helper import rmse
from reader import *
import numpy as np

# ActivityBKT is a non-binary BKT Model
class ActivityBKT():
    def __init__(self, params_dict, kc_list, uniq_student_ids, uniq_activities, activity_learning_progress={}, update_type='by_activity', responsibility='independent'):
        self.n          = len(uniq_student_ids)
        self.num_skills = len(kc_list)
        self.num_acts   = len(uniq_activities)
        self.Q = pd.read_csv('../hotDINA/qmatrix.txt', header=None).to_numpy()
        
        self.timestep = np.zeros((self.n, self.num_skills))
        self.timestep_act = np.zeros((self.n, self.num_acts))
        self.learning_progress = activity_learning_progress
        self.responsibility = responsibility

        self.kc_list    = kc_list
        self.know       = params_dict['know']
        self.guess      = params_dict['guess']
        self.slip       = params_dict['slip']
        self.learn      = params_dict['learn']
        self.forget     = params_dict['forget'] 
        self.know_act   = params_dict['know_act']
        self.guess_act  = params_dict['guess_act']
        self.slip_act   = params_dict['slip_act']
        self.learn_act  = params_dict['learn_act']
        self.forget_act = params_dict['forget_act']

        self.update_type        = update_type 
        self.uniq_student_ids   = uniq_student_ids
        self.uniq_activities    = uniq_activities

    def set_learning_progress(self, student_id, learning_progress, know):
        self.learning_progress[student_id] = learning_progress
        self.know[self.uniq_student_ids.index(student_id)] = know
    
    def update_per_skill(self, observation, i, j):

        t = self.timestep[i][j]
        percent_correct = observation
        percent_incorrect = 1.0 - percent_correct
        prior_know = self.know[i][j]
        prior_not_know = 1.0 - prior_know
        slip = self.slip[i][j]
        no_slip = 1.0 - slip
        guess = self.guess[i][j]
        no_guess = 1.0 - guess
        learn = self.learn[i][j]
        no_learn = 1.0 - learn
        forget = self.forget[i][j]
        no_forget = 1.0 - forget
        posterior_know_given_obs = None
        correct = prior_know * no_slip + prior_not_know * guess
        wrong = 1.0 - correct
        posterior_know_given_obs = (percent_correct * ((prior_know * no_slip) / correct )) + \
                                    (percent_incorrect * ((prior_know * slip) / wrong))
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn
        self.timestep[i][j] = t+1
        self.know[i][j] = posterior_know
    
    def update_per_activity(self, observation, i, j):
        percent_correct = observation
        percent_incorrect = 1.0 - percent_correct
        prior_know = self.know_act[i][j]
        prior_not_know = 1.0 - prior_know
        slip = self.slip_act[i][j]
        no_slip = 1.0 - slip
        guess = self.guess_act[i][j]
        no_guess = 1.0 - guess
        learn = self.learn_act[i][j]
        no_learn = 1.0 - learn
        forget = self.forget_act[i][j]
        no_forget = 1.0 - forget
        posterior_know_given_obs = prior_know
        correct = prior_know * no_slip + prior_not_know * guess
        wrong = 1.0 - correct
        posterior_know_given_obs = (percent_correct * ((prior_know * no_slip) / correct )) + (percent_incorrect * ((prior_know * slip) / wrong))
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn
        self.know_act[i][j] = posterior_know
        
    def update_know_skills(self, activity_name, activity_num, student_num):
        p_know                  = [0] * self.num_skills
        act_counts_per_skill    = [0] * self.num_skills
        student_know_act = self.know_act[student_num]
        for act in self.uniq_activities:
            act_idx = self.uniq_activities.index(act)
            skills = self.Q[act_idx]
            for j in range(self.num_skills):
                skill = skills[j]
                p_know[j] += (student_know_act[act_idx] * skill)
                act_counts_per_skill[j] += skill
        for j in range(self.num_skills):
            if act_counts_per_skill[j] == 0:
                continue
            p_know[j] = p_know[j]/act_counts_per_skill[j]
        self.know[student_num] = p_know
        
    def update(self, activity_observations, student_nums, activities):
        for i in range(len(activities)):
            student_num = student_nums[i]
            if i > 0 and student_num != student_nums[i-1]:
                print("Updating", student_num, "out of", student_nums[-1], "students")
            student_id = self.uniq_student_ids[student_num]
            activity_num = self.uniq_activities.index(activities[i])
            activity_name = activities[i]
            if self.update_type == 'by_activity':
                self.update_per_activity(activity_observations[i], student_num, activity_num)
                self.update_know_skills(activity_name, activity_num, student_num)  
            elif self.update_type == 'by_skill':
                skills = self.Q[activity_num]
                for j in range(len(skills)):
                    if skills[j] == 1:
                        self.update_per_skill(activity_observations[i], student_num, j)
            
            if student_id not in self.learning_progress:
                self.learning_progress[student_id] = [self.know[student_num].tolist()]
            else:
                self.learning_progress[student_id].append(self.know[student_num].tolist())

    def predict_percent_corrects(self, student_ids, skills, actual_observations=None):
        correct_preds = []
        min_correct_preds = []
        n = len(student_ids)

        for k in range(n):
            student_id = student_ids[k]
            student_num = self.uniq_student_ids.index(student_id)
            correct = 1.0
            min_correct = 1.0
            for skill_num in skills[k]:
                p_know = self.know[student_num][skill_num]
                p_not_know = 1.0 - p_know
                p_guess = self.guess[student_num][skill_num]
                p_slip = self.slip[student_num][skill_num]
                p_not_slip = 1.0 - p_slip
                # Independent subskills prediction
                correct = correct * ((p_know * p_not_slip) + (p_not_know * p_guess))
                # Weakest-subskill performance prediction 
                min_correct = min(min_correct, (p_know * p_not_slip) + (p_not_know * p_guess))
            
            correct_preds.append(correct)
            min_correct_preds.append(min_correct)
        
        return correct_preds, min_correct_preds

    def simulate_response(self, student_id, skill, types='normal'):
        correct_preds, min_correct_preds = self.predict_percent_corrects([student_id], [skill])
        if types == 'normal':
            correct_prob = correct_preds[0]
        elif types == 'hardest_skill':
            correct_prob = min_correct_preds[0]
        # Sample response from Bern(correct_prob)
        response = np.random.binomial(n=1, p=correct_prob)
        return response

    def predict_p_correct(self, student_num, skill, update=False):
        i = student_num
        correct = 1.0

        for j in skill:
            p_know = self.know[i][j]
            p_not_know = 1.0 - p_know
            p_guess = self.guess[i][j]
            p_slip = self.slip[i][j]
            p_not_slip = 1.0 - p_slip
            if self.responsibility == 'independent':
                correct = correct * ((p_know * p_not_slip) + (p_not_know * p_guess))
            elif self.responsibility == 'blame_weakest':
                correct = min(correct, (p_know * p_not_slip) + (p_not_know * p_guess))
        
        if update == True:
            self.update([correct], [student_num], [activity])

        return correct