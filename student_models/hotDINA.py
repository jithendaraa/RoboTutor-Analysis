import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from helper import sigmoid
import pickle
import matplotlib.pyplot as plt
import math

# GLOBALLY CONSTANT FOR ANY VILLAGE
Q = pd.read_csv('../../hotDINA/qmatrix.txt', header=None).to_numpy()
J = 1712
K = 22
village_num = "115"
NUM_ENTRIES = 1000

class hotDINA_skill():
    def __init__(self, I, K, T, params_skill_dict):
        
        self.I = I
        self.K = K
        self.T = T

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
        for i in range(I):
            self.knows[i] = []
            self.avg_knows[i] = []

        self.knews  = np.zeros((I, K))

        # Insert "knews" as knows@t=0
        for i in range(I):
            for k in range(K):
                self.knews[i][k]    = sigmoid(1.7 * self.a[k] * (self.theta[i] - self.b[k]))
            
            self.knows[i].append(self.knews[i].tolist())
            self.avg_knows[i].append(np.mean(self.knews[i]))

    def update_skill(self, i, k, t, y, know):
        prior_know = self.knows[i][-1][k]
        posterior_know = None
        p_correct = (self.ss[k] * prior_know) + (self.g[k] * (1 - prior_know)) 

        if y == 1:
            posterior_know = self.ss[k] * prior_know/p_correct
        elif y == 0:
            posterior_know = self.g[k] * (1 - prior_know) / p_correct

        posterior_know = posterior_know + (1-posterior_know) * self.learn[k]    

        know[k] = posterior_know
        return know

    def update(self, observations, items, users):

        for i in range(len(observations)):
            user = users[i]
            correct = int(observations[i])
            item = items[i]
            skills = Q[item]
            know = self.knows[user][-1]
            print(know)
            for k in range(len(skills)):
                skill = skills[k]
                if skill == 1:
                    know = self.update_skill(user, k, i, correct, know)
            
            self.knows[user].append(know)
            self.avg_knows[user].append(np.mean(know))

    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y)
        plt.show()

class hotDINA():
    def __init__(self, I, J, K, T, params_dict, update_type="bayesian"):
        
        self.I = I
        self.J = J
        self.K = K
        self.T = T

        self.update_type = update_type

        # Skills for hotDINA_params. Theta Ix1 vector, guess and slip are Jx1, the rest are Kx1
        self.theta  = params_dict['theta']
        self.a      = params_dict['a']
        self.b      = params_dict['b']
        self.learn  = params_dict['learn']
        self.g      = params_dict['g']
        self.ss     = params_dict['ss']

        # P(Know)'s of every student for every skill after every opportunity
        self.eta = {}
        self.alpha = np.zeros((I, K))
        self.avg_knows = {}
        
        for i in range(I):
            self.eta[i] = []
            self.avg_knows[i] = []

        # Insert "knews" as knows@t=0
        for i in range(I):
            for k in range(K):
                self.alpha[i][k]    = sigmoid(1.7 * self.a[k] * (self.theta[i] - self.b[k]))
        
        for i in range(I):
            know = [1.0] * J
            for j in range(J):
                for k in range(K):
                    know[j] = pow(self.alpha[i][k], Q[j][k]) * know[j]
            
            self.eta[i].append(know)
            self.avg_knows[i].append(np.mean(know))

    def update_skill(self, i, j, k, t, y, know):
        
        prior_know = self.eta[i][-1][k]
        posterior_know = None
        p_correct = (self.ss[j] * prior_know) + (self.g[j] * (1 - prior_know)) 

        if y == 1:
            posterior_know = self.ss[j] * prior_know/p_correct
        elif y == 0:
            posterior_know = self.g[j] * (1 - prior_know) / p_correct

        # posterior_know = posterior_know + (1-posterior_know) * self.learn[k]    

        know[k] = posterior_know
        return know

    def update(self, observations, items, users):

        if self.update_type == "bayesian":
            for i in range(len(observations)):
                user = users[i]
                correct = int(observations[i])
                item = items[i]
                skills = Q[item]
                know = self.eta[user][-1]
                for k in range(len(skills)):
                    skill = skills[k]
                    if skill == 1:
                        know = self.update_skill(user, item, k, i, correct, know)

                self.eta[user].append(know)
                self.avg_knows[user].append(np.mean(know))
        
        else if self.update_type == "hardest_skill":
            pass
            # self.avg_knows[user].append(np.mean(know))

    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y)
        plt.show()



with open('../../hotDINA/pickles/data/data' + village_num + '_' + str(NUM_ENTRIES) + '.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)

T = data_dict['T']
I = len(T)

params_skill_dict = {
    'theta' :   np.ones(I),
    'a'     :   np.ones(K),
    'b'     :   np.ones(K),
    'learn' :   np.ones(K),
    'g'     :   np.ones(K),
    'ss'    :   np.ones(K)      
}

params_dict = {
    'theta' :   np.ones(I),
    'a'     :   np.ones(K),
    'b'     :   np.ones(K),
    'learn' :   np.ones(K),
    'g'     :   np.ones(J),
    'ss'    :   np.ones(J)      
}

observations, items, users = data_dict['y'], data_dict['items'], data_dict['users']

# model = hotDINA_skill(I, K, T, params_skill_dict)
# model.update(observations, items, users)
# model.plot_avg_knows()

model = hotDINA(I, J, K, T, params_dict)
model.update(observations, items, users)

# data = pickle.load( open( "../../hotDINA/pickles/model_fit_" + village_num + "_" + str(NUM_ENTRIES) + ".pkl", "rb" ) )
# fit = data['fit']
# print(fit)