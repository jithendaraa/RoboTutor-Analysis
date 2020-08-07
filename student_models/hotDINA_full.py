import numpy as np
import matplotlib.pyplot as plt
import math


# GLOBALLY CONSTANT FOR ANY VILLAGE
Q = pd.read_csv('../../hotDINA/qmatrix.txt', header=None).to_numpy()
J = 1712
K = 22

class hotDINA_full():
    def __init__(self, params_dict, T, , update_type="bayesian"):
        
        self.I = len(params_dict['theta'])
        self.J = len(params_dict['g'])
        self.K = len(params_dict['a'])
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
        p_wrong = 1.0 - p_correct

        if y == 1:
            posterior_know = self.ss[j] * prior_know/p_correct
        elif y == 0:
            posterior_know = prior_know * (1 - self.ss[j]) / p_wrong

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
        
        elif self.update_type == "hardest_skill":
            pass
            # self.avg_knows[user].append(np.mean(know))

    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y)
        plt.show()
