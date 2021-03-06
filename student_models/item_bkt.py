import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from helper import *

# ITEM BKT is a binary BKT Model
class ItemBKT:
    # value of knew is always knew @ time (timestep)
    # g -> guess, s->slip, l->learn, f->forget, k->know: all of these are n*u arrays where we have n students and u skills or items
    # timestep-> n*u array which tells the current timestep of student n for skill u
    # KC subtest -> skills as per CTA table in 'Data/CTA.xlsx'
    """
        Functions
        ---------
        __init__()            : Initialises the class and sets class attributes
        update_per_obs()      : Does the step wise BKT update for "one observation"
        update()              : Does step wise BKT update for some number of observations by calling the update_per_obs() function
        predict_p_correct()   : After fitting, this returns P(Correct) of getting an item correct based on learnt BKT params 
    """
    def __init__(self, params_dict, kc_list, uniq_student_ids, update_type='independent'):
        """
            Variables
            ---------
            self.know       --> an (n x u) matrix where self.know[i][j] is P(K @0) or prior know before any opportunities of student i, for skill j
            self.guess      --> an (n x u) matrix where self.guess[i][j] is P(Guess) of student i, for skill j. For an item with 4 choices this could be 0.25
            self.slip       --> an (n x u) matrix where self.slip[i][j] is P(Slip) of student i, for skill j. 
            self.learn      --> an (n x u) matrix where self.learn[i][j] is P(Learn) or P(Transit) of student i, for skill j. A fast learner will have a high P(learn)
            self.forget     --> an (n x u) matrix where self.foget[i][j] is P(Forget) of student i, for skill j. Usually always assumed to be 0.
            kc_list         --> list (of length num_skills) of all skills according to the CTA table 
        """
        self.update_type = update_type
        self.know = params_dict['know']
        self.guess = params_dict['guess']
        self.slip = params_dict['slip']
        self.learn = params_dict['learn']
        self.forget = params_dict['know']
        
        self.kc_list = kc_list
        self.n = len(uniq_student_ids)
        self.u = len(kc_list)
        self.students = uniq_student_ids
        self.learning_progress = {}

        for i in range(self.n):
            student_id = self.students[i]
            self.learning_progress[student_id] = [self.know[i].copy()]

    def update_per_obs(self, observation, i, j):
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
        correct = 10e-8 + (prior_know * no_slip) + (prior_not_know * guess)
        wrong = 1.0 - correct
        if observation == 1:
            posterior_know_given_obs = (prior_know * no_slip / correct)
        elif observation == 0:
            posterior_know_given_obs = (prior_know * slip / wrong)
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn
        self.know[i][j] = posterior_know

    def update(self, student_nums, skill_nums, corrects):
        num_rows = len(student_nums)
        for i in range(num_rows):
            student_num = student_nums[i]
            student_id = self.students[student_num]
            for skill_num in skill_nums[i]:
                self.update_per_obs(corrects[i], student_num, skill_num)
            self.learning_progress[student_id].append(self.know[student_num].copy())

    def predict_p_correct(self, student_num, skill, update=False):
        i = student_num
        correct = 1.0
        
        for j in skill:
            p_know = self.know[i][j]
            p_not_know = 1.0 - p_know
            p_guess = self.guess[i][j]
            p_slip = self.slip[i][j]
            p_not_slip = 1.0 - p_slip
            if self.update_type == 'independent':
                correct = correct * ((p_know * p_not_slip) + (p_not_know * p_guess))
            elif self.update_type == 'blame_weakest':
                correct = min(correct, (p_know * p_not_slip) + (p_not_know * p_guess))
        
        student_response = None
        if update == True:
            if correct > 1 or correct < 0:
                print(correct)
            student_response = np.random.binomial(1, correct)
            self.update([student_num], [skill], [student_response])

        return correct, student_response

    def get_rmse(self, student_nums, skills, observations, train_ratio=0.5):
        predicted_responses  = []
        predicted_p_corrects = []
        idxs = []
        corrects = []
        unique_student_nums = pd.unique(np.array(student_nums)).tolist()
        for i in range(len(student_nums)):
            student_num = student_nums[i]
            if i > 0 and student_num != student_nums[i-1]:
                idxs.append(i)
        for i in range(len(idxs)):
            if i == 0:
                skills_         = skills[0:idxs[i]]
                student_nums_   = student_nums[0:idxs[i]] 
                corrects_       = observations[0:idxs[i]]
            else:
                skills_         = skills[idxs[i-1]:idxs[i]]
                student_nums_   = student_nums[idxs[i-1]:idxs[i]]
                corrects_       = observations[idxs[i-1]:idxs[i]]
            entries = len(skills_) 
            test_idx = int(train_ratio * entries)
            # Update on training data
            self.update(student_nums_[:test_idx], skills_[:test_idx], corrects_[:test_idx])
            # Test accuracy on test data
            _student_nums   = student_nums_[test_idx:]
            _skills         = skills_[test_idx:]
            _corrects       = corrects_[test_idx:]

            if len(corrects) == 0:
                corrects = _corrects.copy()
            else:
                corrects = corrects + _corrects

            for j in range(len(_student_nums)):
                student_num = _student_nums[j]
                skill       = _skills[j]
                correct, student_response = self.predict_p_correct(student_num, skill, update=True)
                
                predicted_p_corrects.append(correct)
                predicted_responses.append(student_response)
        
        if len(idxs) > 0:
            skills_         = skills[idxs[-1]:]
            student_nums_   = student_nums[idxs[-1]:]
            corrects_       = observations[idxs[-1]:]
        else:
            skills_         = skills
            student_nums_   = student_nums
            corrects_       = observations
        
        entries = len(skills_) 
        test_idx = int(train_ratio * entries)
        # Update training data
        self.update(student_nums_[:test_idx], skills_[:test_idx], corrects_[:test_idx])
        # Test data
        _student_nums   = student_nums_[test_idx:]
        _skills         = skills_[test_idx:]
        _corrects       = corrects_[test_idx:]

        if len(corrects) == 0:
            corrects = _corrects.copy()
        else:
            corrects = corrects + _corrects
        
        for j in range(len(_student_nums)):
            student_num = _student_nums[j]
            skill       = _skills[j]
            correct, student_response = self.predict_p_correct(student_num, skill, update=True)
            
            predicted_p_corrects.append(correct)
            predicted_responses.append(student_response)

        majority_response   = [1.0] * len(corrects)
        prob_rmse           = rmse(predicted_p_corrects, corrects)
        sampled_rmse        = rmse(predicted_responses, corrects)
        majority_class_rmse = rmse(majority_response, corrects)
        return prob_rmse, sampled_rmse, majority_class_rmse, predicted_responses, predicted_p_corrects