import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import math
import os
from helper import *

class hotDINA_skill():
    def __init__(self, params_dict, path_to_Qmatrix):
        
        self.I = len(params_dict['theta'])
        self.K = len(params_dict['a'])        
        self.Q = pd.read_csv(path_to_Qmatrix, header=None).to_numpy()
        
        # Skills for hotDINA_params. Theta Ix1 vector, the rest are Kx1
        self.theta  = params_dict['theta']
        self.a      = params_dict['a']
        self.b      = params_dict['b']
        self.learn  = params_dict['learn']
        self.g      = params_dict['g']
        self.ss     = params_dict['ss']

        # P(Know)'s of every student for every skill after every opportunity
        self.knows  = {}
        self.avg_knows = {}
        for i in range(self.I):
            self.knows[i] = []
            self.avg_knows[i] = []

        self.knews  = np.zeros((self.I, self.K))
        self.bayesian_update = True

        # Insert "knews" as knows@t=0
        for i in range(self.I):
            for k in range(self.K):
                self.knews[i][k]    = sigmoid(1.7 * self.a[k] * (self.theta[i] - self.b[k]))
            
            self.knows[i].append(self.knews[i].tolist())
            self.avg_knows[i].append(np.mean(self.knews[i]))

    def update_skill(self, i, k, t, y, know):
        prior_know = know[k]
        posterior_know = None
        p_correct = (self.ss[k] * prior_know) + (self.g[k] * (1 - prior_know)) 
        p_wrong = 1.0 - p_correct
        
        if p_wrong == 0 or p_correct == 0:
            print("oof")
        elif prior_know == 0.0 or self.ss[k] == 0:
            print("oof1")

        if self.bayesian_update:
            if y == 1:
                posterior_know = self.ss[k] * prior_know/p_correct
            elif y == 0:
                posterior_know = (1 - self.ss[k]) * prior_know / p_wrong
        
            print(posterior_know)
            posterior_know = posterior_know + (1-posterior_know) * self.learn[k]    
        
        else:
            posterior_know = prior_know + (1-prior_know) * self.learn[k]

        know[k] = posterior_know
        return know

    def update(self, observations, items, users, bayesian_update=True, plot=True):
        
        self.bayesian_update = bayesian_update

        for i in range(len(observations)):
            user = users[i]
            correct = int(observations[i])
            item = items[i]
            skills = self.Q[item]
            know = self.knows[user][-1]
            for k in range(len(skills)):
                skill = skills[k]
                if skill == 1:
                    know = self.update_skill(user, k, i, correct, know)
            
            self.knows[user].append(know)
            self.avg_knows[user].append(np.mean(know))

        if plot:
            for i in range(self.I):
                y = self.avg_knows[i]
                x = np.arange(len(y)).tolist()
                if bayesian_update:
                    plt.plot(x, y, label="student_" + str(i) + "_bayesian" )
                else:
                    plt.plot(x, y, label="student_" + str(i) + "_no_bayesian" )

    def predict_response(self, item, user, update=False, bayesian_update=True, plot=False):

        current_know = self.knows[user][-1]
        skills = self.Q[item]
        p_correct = 1.0

        for k in len(skills):
            skill = skills[k]
            if skill == 1:
                p_correct_skill = (current_know[k] * self.ss[item]) + ((1 - current_know[k]) * self.g[item])
                p_correct = p_correct * p_correct_skill
        
        # response = Bern(p_correct)
        predicted_response = int(np.random.binomial(n=1, p=p_correct))

        if update:
            self.update([predicted_response], [item], [user], bayesian_update, plot)
        
        return predicted_response
    
    def predict_responses(self, items, users, bayesian_update=True, plot=False, observations=None):
        
        predicted_responses = []

        for i in range(len(users)):
            user = users[i]
            item = items[i]
            predicted_response = self.predict_response(item, user, update=True, bayesian_update=bayesian_update, plot=plot)
            predicted_responses.append(predicted_response)
        
        if observations != None:
            accuracy = int(np.sum(np.abs(np.array(observations) - np.array(predicted_responses)))) * 100 /len(observations)
            print("Accuracy: ", accuracy, "%")

        return predicted_responses

    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y, label="student" + str(i))
