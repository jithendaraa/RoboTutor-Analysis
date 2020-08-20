import sys
sys.path.append('..')

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from pathlib import Path
import pickle

from helper import *
from reader import *

class hotDINA_full():
    def __init__(self, params_dict, path_to_Qmatrix, responsibilty='independent'):

        params_dict['g'] = 0.25 * np.ones((1712))
        params_dict['ss'] = 0.85 * np.ones((1712))
        
        self.I = len(params_dict['theta'])
        self.J = len(params_dict['g'])
        self.K = len(params_dict['a'])
        self.Q = pd.read_csv(path_to_Qmatrix, header=None).to_numpy()
        self.responsibilty = responsibilty
        # Skills for hotDINA_params. Theta Ix1 vector, guess and slip are Jx1, the rest are Kx1
        self.theta  = params_dict['theta']
        self.a      = params_dict['a']
        self.b      = params_dict['b']
        self.learn  = params_dict['learn']
        self.g      = params_dict['g']
        self.ss     = params_dict['ss']
        # P(Know)'s of every student for every skill after every opportunity
        self.eta = {}
        self.alpha = {}
        self.avg_knows = {}
        for i in range(self.I):
            self.eta[i] = []
            self.avg_knows[i] = []
            self.alpha[i] = []
         # Insert "knews" as knows@t=0
        for i in range(self.I):
            alpha = np.zeros((self.K)).tolist()
            for k in range(self.K):
                alpha[k] = 0.1
                # alpha[k] = sigmoid(1.7 * self.a[k] * (self.theta[i] - self.b[k]))
            self.alpha[i].append(alpha)
        
        for i in range(self.I):
            self.avg_knows[i].append(np.mean(self.alpha[i][-1]))

    def update(self, observations, items, users, train_students=None):
        for i in range(len(observations)):
            user = users[i]
            j = items[i]
            if train_students != None and user not in train_students:
                continue
            correct = int(observations[i])
            skills = self.Q[j]
            alpha = self.alpha[user][-1].copy()
            eta = 1.0
            
            for k in range(len(skills)):
                eta = eta * pow(alpha[k], skills[k])
            
            if self.responsibilty == "independent":
                for k in range(len(skills)):
                    p_correct = (eta * self.ss[j]) + (self.g[j] * (1-eta))
                    p_wrong = 1.0 - p_correct
                    if correct == 1:
                        eta_given_obs = self.ss[j] * eta/ p_correct
                    else:
                        eta_given_obs = (1 - self.ss[j]) * eta/ p_wrong
                    if skills[k] == 1:
                        alpha[k] = eta_given_obs
                        alpha[k] = alpha[k] + (1-alpha[k]) * self.learn[k]
            
            self.alpha[user].append(alpha)
            self.avg_knows[user].append(np.mean(alpha))

    def predict_response(self, item, user, update=True):
        j = item
        skills = self.Q[j]
        eta = 1.0
        alpha = self.alpha[user][-1].copy()
        
        for k in range(self.K):
            eta = eta * pow(alpha[k], skills[k])

        predicted_p_correct = (self.ss[j] * eta) + (self.g[j] * (1 - eta))
        predicted_response = np.random.binomial(n=1, p=predicted_p_correct)

        if update:
            self.update([predicted_response], [item], [user])

        return predicted_p_correct, predicted_response
    
    def get_rmse(self, users, items, observations, train_ratio=0.5):
        predicted_responses = []
        predicted_p_corrects = []
        idxs = []
        corrects = []
        unique_users = pd.unique(np.array(users)).tolist()

        for i in range(len(users)):
            user = users[i]
            if i > 0 and user != users[i-1]:
                idxs.append(i)
        
        for i in range(len(idxs)):
            if i == 0:
                items_ = items[0:idxs[i]]
                users_ = users[0:idxs[i]]
                obs_ = observations[0:idxs[i]]
            else:
                items_ = items[idxs[i-1]:idxs[i]]
                users_ = users[idxs[i-1]:idxs[i]]
                obs_ = observations[idxs[i-1]:idxs[i]]
            entries = len(items_)
            test_idx = int(train_ratio * entries)
            # Update on training data
            self.update(obs_[:test_idx], items_[:test_idx], users_[:test_idx])
            # Test accuracy on test data
            _users = users_[test_idx:]
            _items = items_[test_idx:]
            _obs = obs_[test_idx:]
            if len(corrects) == 0:
                corrects = _obs.copy()
            else:
                corrects = corrects + _obs
            for j in range(len(_users)):
                user = _users[j]
                item = _items[j]
                predicted_p_correct, predicted_response = self.predict_response(item, user, update=True)
                if self.responsibilty == 'independent':
                    predicted_responses.append(predicted_response)
                    predicted_p_corrects.append(predicted_p_correct)

        if len(idxs) > 0:
            items_ = items[idxs[-1]:]
            users_ = users[idxs[-1]:]
            obs_ = observations[idxs[-1]:]
        else:
            items_ = items
            users_ = users
            obs_ = observations
        
        entries = len(items_) 
        test_idx = int(train_ratio * entries)
        # Update training data
        self.update(obs_[:test_idx], items_[:test_idx], users_[:test_idx])
        # Test data
        _users = users_[test_idx:]
        _items = items_[test_idx:]
        _obs = obs_[test_idx:]

        if len(corrects) == 0:
            corrects = _obs.copy()
        else:
            corrects = corrects + _obs
            
        for j in range(len(_users)):
            user = _users[j]
            item = _items[j]
            predicted_p_correct, predicted_response = self.predict_response(item, user, update=True)
            if self.responsibilty == 'independent':
                predicted_responses.append(predicted_response)
                predicted_p_corrects.append(predicted_p_correct)

        majority_response = [1.0] * len(corrects)
        majority_class_rmse = rmse(majority_response, corrects)
        predicted_responses_rmse = rmse(predicted_responses, corrects)
        predicted_p_corrects_rmse = rmse(predicted_p_corrects, corrects)

        return majority_class_rmse, predicted_responses_rmse, predicted_p_corrects_rmse

    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y)
        plt.show()

if __name__ == '__main__':
    
    I = 8
    J = 1712
    K = 22 

    dummy_params_dict = {
        'theta' : (-np.log(9)/1.7) * np.ones((I, 1)),
        'a'     : np.ones((K, 1)),
        'b'     : 0 * np.ones((K, 1)),
        'learn' : 0.5 * np.ones((K, 1)),
        'g'     : 0.25 * np.ones((J, 1)),
        'ss'    : 0.85 *np.ones((J, 1))
    }

    path_to_Qmatrix = '../../hotDINA/qmatrix.txt'

    village = '130'
    obs = '10000'

    path_to_data_file = os.getcwd() + '/../../hotDINA/pickles/data/data'+ village + '_' + obs +'.pickle'
    data_file = Path(path_to_data_file)
    if data_file.is_file() == False:
        # if data_file does not exist, get it
        os.chdir('../../hotDINA')
        get_data_file_command = 'python get_data_for_village_n.py -v ' + village + ' -o ' + obs
        os.system(get_data_file_command)
        os.chdir('../RoboTutor-Analysis/student_models')

    os.chdir('../../hotDINA')
    with open(path_to_data_file, 'rb') as handle:
        data_dict = pickle.load(handle)
    os.chdir('../RoboTutor-Analysis/student_models')

    observations    = data_dict['y']
    items           = data_dict['items']
    users           = data_dict['users']
    
    xs = []
    avg_majority_class_rmses, avg_predicted_responses_rmses, avg_predicted_p_corrects_rmses = [], [], []
    majority_class_rmses, predicted_responses_rmses, predicted_p_corrects_rmses = [], [], []
    
    for _ in range(1):
        rmse_vals = []
        majority_class_rmses = []
        xs = []
        for i in np.linspace(0,0.9,10):
            model = hotDINA_full(dummy_params_dict, path_to_Qmatrix)
            majority_class_rmse, predicted_responses_rmse, predicted_p_corrects_rmse = model.get_rmse(users, items, observations, train_ratio=i)
            
            majority_class_rmses.append(majority_class_rmse)
            predicted_responses_rmses.append(predicted_responses_rmse)
            predicted_p_corrects_rmses.append(predicted_p_corrects_rmse)
            xs.append(i*len(observations))

        avg_majority_class_rmses.append(majority_class_rmses)
        avg_predicted_responses_rmses.append(predicted_responses_rmses)
        avg_predicted_p_corrects_rmses.append(predicted_p_corrects_rmses)

    avg_majority_class_rmses = np.mean(avg_majority_class_rmses, axis=0)
    avg_predicted_responses_rmses = np.mean(avg_predicted_responses_rmses, axis=0)
    avg_predicted_p_corrects_rmses = np.mean(avg_predicted_p_corrects_rmses, axis=0)

    plt.plot(xs, avg_predicted_p_corrects_rmses, 'g', label='p_response RMSE')
    plt.plot(xs, avg_predicted_responses_rmses, 'r', label='response RMSE')
    plt.plot(xs, avg_majority_class_rmses, 'black', label='Majority class RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('Number of attempts')
    plt.legend()
    plt.show()

