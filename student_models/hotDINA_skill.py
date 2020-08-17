from helper import *
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('..')


class hotDINA_skill():
    def __init__(self, params_dict, path_to_Qmatrix, responsibilty='independent'):

        self.I = len(params_dict['theta'])
        self.K = len(params_dict['a'])
        self.Q = pd.read_csv(path_to_Qmatrix, header=None).to_numpy()

        # Skills for hotDINA_params. Theta Ix1 vector, the rest are Kx1
        self.theta = params_dict['theta']
        self.a = params_dict['a']
        self.b = params_dict['b']
        self.learn = params_dict['learn']
        self.g = params_dict['g']
        self.ss = params_dict['ss']

        self.bayesian_update = True
        self.responsibilty = responsibilty
        # P(Know)'s of every student for every skill after every opportunity
        self.alpha = {}
        self.avg_knows = {}
        for i in range(self.I):
            self.alpha[i] = []
            self.avg_knows[i] = []
        self.knews = np.zeros((self.I, self.K))
        # Insert "knews" as knows@t=0
        for i in range(self.I):
            for k in range(self.K):
                self.knews[i][k] = sigmoid(1.7 * self.a[k] * (self.theta[i] - self.b[k]))
            self.alpha[i].append(self.knews[i].tolist())
            self.avg_knows[i].append(np.mean(self.knews[i]))

    def update_skill(self, i, k, t, y, know):
        prior_know = know[k]
        posterior_know = None
        p_correct = (self.ss[k] * prior_know) + (self.g[k] * (1 - prior_know))
        p_wrong = 1.0 - p_correct

        if self.bayesian_update:
            if y == 1:
                posterior_know = self.ss[k] * prior_know/p_correct
            elif y == 0:
                posterior_know = (1 - self.ss[k]) * prior_know / p_wrong

            posterior_know = posterior_know + \
                (1-posterior_know) * self.learn[k]

        else:
            posterior_know = prior_know + (1-prior_know) * self.learn[k]

        know[k] = posterior_know
        return know

    def update(self, observations, items, users, bayesian_update=True, plot=True, train_students=None):
        self.bayesian_update = bayesian_update
        for i in range(len(observations)):
            user = users[i]
            if train_students != None and user not in train_students:
                continue
            correct = int(observations[i])
            item = items[i]
            skills = self.Q[item]
            know = self.alpha[user][-1]
            for k in range(len(skills)):
                skill = skills[k]
                if skill == 1:
                    know = self.update_skill(user, k, i, correct, know)
            self.alpha[user].append(know)
            self.avg_knows[user].append(np.mean(know))
        if plot:
            for i in range(self.I):
                y = self.avg_knows[i]
                x = np.arange(len(y)).tolist()
                if bayesian_update:
                    plt.plot(x, y, label="student_" + str(i) + "_bayesian")
                else:
                    plt.plot(x, y, label="student_" + str(i) + "_no_bayesian")
            plt.legend()
            plt.show()

    def predict_response(self, item, user, update=False, bayesian_update=True, plot=False):
        current_know = self.alpha[user][-1]
        skills = self.Q[item]
        p_correct = 1.0
        p_min_correct = 1.0
        for k in range(len(skills)):
            skill = skills[k]
            if skill == 1:
                p_correct_skill = (
                    current_know[k] * self.ss[k]) + ((1 - current_know[k]) * self.g[k])
                p_correct = p_correct * p_correct_skill
                p_min_correct = min(p_min_correct, p_correct_skill)
        correct_response = int(np.random.binomial(n=1, p=p_correct))
        min_correct_response = int(np.random.binomial(n=1, p=p_min_correct))
        
        if update:
            if self.responsibilty == 'independent':
                self.update([correct_response], [item], [user], bayesian_update, plot)
            elif self.responsibilty == 'blame_weakest':
                self.update([min_correct_response], [item], [user], bayesian_update, plot)

        return correct_response, min_correct_response

    def predict_responses(self, items, users, bayesian_update=True, plot=False, observations=None):

        predicted_responses = []

        for i in range(len(users)):
            user = users[i]
            item = items[i]
            predicted_response = self.predict_response(
                item, user, update=True, bayesian_update=bayesian_update, plot=plot)
            predicted_responses.append(predicted_response)

        if observations != None:
            accuracy = int(np.sum(np.abs(np.array(
                observations) - np.array(predicted_responses)))) * 100 / len(observations)
            print("Accuracy: ", accuracy, "%")
        return predicted_responses

    def get_rmse(self, users, items, observations, train_ratio=0.5):
        predicted_responses = []
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
            self.update(obs_[:test_idx], items_[:test_idx], users_[:test_idx], plot=False)
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
                correct_response, min_correct_response = self.predict_response(item, user, update=True)
                if self.responsibilty == 'independent':
                    predicted_responses.append(correct_response)
                elif self.responsibilty == 'blame_weakest':
                    predicted_responses.append(min_correct_response)

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
        self.update(obs_[:test_idx], items_[:test_idx], users_[:test_idx], plot=False)
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
            correct_response, min_correct_response = self.predict_response(item, user, update=True)
            if self.responsibilty == 'independent':
                predicted_responses.append(correct_response)
            elif self.responsibilty == 'blame_weakest':
                predicted_responses.append(min_correct_response)

        majority_response = [1.0] * len(corrects)
        majority_class_rmse = rmse(majority_response, corrects)
        rmse_val = rmse(predicted_responses, corrects)
        return rmse_val, majority_class_rmse

    def plot_avg_knows(self):
        for i in range(self.I):
            y = self.avg_knows[i]
            x = np.arange(len(y)).tolist()
            plt.plot(x, y, label="student" + str(i))


if __name__ == '__main__':

    from pathlib import Path
    import pickle

    path = os.getcwd() + '/../slurm_outputs'
    village = '130'
    observations = '1000'
    params_dict = slurm_output_params(path, village)

    path_to_Qmatrix = os.getcwd() + '/../../hotDINA/qmatrix.txt'

    path_to_data_file = os.getcwd() + '/../../hotDINA/pickles/data/data' + \
        village + '_' + observations + '.pickle'
    data_file = Path(path_to_data_file)

    if data_file.is_file() == False:
        # if data_file does not exist, get it
        os.chdir('../../hotDINA')
        get_data_file_command = 'python get_data_for_village_n.py -v ' + \
            village + ' -o ' + observations
        os.system(get_data_file_command)
        os.chdir('../RoboTutor-Analysis/student_models')

    os.chdir('../../hotDINA')
    with open(path_to_data_file, 'rb') as handle:
        data_dict = pickle.load(handle)
    os.chdir('../RoboTutor-Analysis/student_models')

    # update(self, observations, items, users, bayesian_update=True, plot=True)

    observations = data_dict['y']
    items = data_dict['items']
    users = data_dict['users']

    model = hotDINA_skill(params_dict, path_to_Qmatrix)
    model.update(observations, items, users, bayesian_update=True, plot=True)

    model2 = hotDINA_skill(params_dict, path_to_Qmatrix)
    model2.update(observations, items, users, bayesian_update=False, plot=True)
    plt.show()
