import numpy as np
import pandas as pd
import sys
sys.path.append('..')
import pickle
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path

from helper import *

# GLOBALLY CONSTANT FOR ANY VILLAGE
Q = pd.read_csv('../../hotDINA/qmatrix.txt', header=None).to_numpy()
J = 1712
K = 22

slurm_params_files = {
    "130":  "10301619"
}

class hotDINA_skill():
    def __init__(self, params_dict, T):
        
        self.I = len(params_dict['theta'])
        self.K = len(params_dict['a'])
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
        self.bayesian_update = True

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

        if self.bayesian_update:
            if y == 1:
                posterior_know = self.ss[k] * prior_know/p_correct
            elif y == 0:
                posterior_know = self.g[k] * (1 - prior_know) / p_correct
            
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
            skills = Q[item]
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

    # def predict(self, item, user):


    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y, label="student" + str(i))


if __name__ == "__main__":
    village_num = "130"
    NUM_ENTRIES = "1000"

    path_to_data_file = os.getcwd() + '/../../hotDINA/pickles/data/data' + village_num + '_' + str(NUM_ENTRIES) + '.pickle'

    if Path(path_to_data_file).is_file() == False:
        # do something
        os.chdir('../../hotDINA')
        get_data_for_village_command = "python get_data_for_village_n.py -v " + village_num + " -o " + NUM_ENTRIES
        os.system(get_data_for_village_command)
        os.chdir("../RoboTutor-Analysis")
    
    with open(path_to_data_file, 'rb') as handle:
        data_dict = pickle.load(handle)

    T = data_dict['T']
    I = len(T)
    observations, items, users = data_dict['y'], data_dict['items'], data_dict['users']

    output_file_path = os.getcwd() + "/../slurm_outputs"
    params_skill_dict = slurm_output_params(path=output_file_path, village=village_num,slurm_id=slurm_params_files[village_num])

    model = hotDINA_skill(params_skill_dict, T)
    model.update(observations, items, users, True, True)

    model2 = hotDINA_skill(params_skill_dict, T)
    model2.update(observations, items, users, False, False)

    plt.legend()
    plt.show()
