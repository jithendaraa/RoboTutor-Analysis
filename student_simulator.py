# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from pathlib import Path

from reader import *
from helper import *

from student_models.item_bkt import ItemBKT
from student_models.activity_bkt import ActivityBKT
from student_models.hotDINA_skill import hotDINA_skill
from student_models.hotDINA_full import hotDINA_full

class StudentSimulator():
    def __init__(self, village="114", observations="10000", student_model_name="ItemBKT", subscript="student_specific", path='', new_student_params=None):
        self.CONSTANTS = {
            "PATH_TO_CTA"                           : "Data/CTA.xlsx",
            "PATH_TO_ACTIVITY_TABLE"                : "Data/Activity_table_v4.1_22Apr2020.pkl",
            "VILLAGE"                               : village,
            "OBSERVATIONS"                          : observations,
            "SUBSCRIPT"                             : subscript, #"student_specific" or "village_specific" for BKT subscripting
            "PRINT_PARAMS_FOR_STUDENT"              : True,
            "PATH_TO_VILLAGE_STEP_TRANSAC_FILES"    : [],
            "STUDENT_MODEL_INITIALISER"             : {
                                                        "ItemBKT"       : "self.student_model = ItemBKT(self.params_dict, self.kc_list, self.uniq_student_ids)",
                                                        "ActivityBKT"   : "self.student_model = ActivityBKT(self.params_dict, self.kc_list, self.uniq_student_ids, self.uniq_activities, self.activity_learning_progress, path=path)",
                                                        "hotDINA_skill" : "self.student_model = hotDINA_skill(self.params_dict, path_to_Qmatrix)",
                                                        "hotDINA_full"  : "self.student_model = hotDINA_full(self.params_dict, path_to_Qmatrix)"
                                                    },
        }
        path_to_Qmatrix = os.getcwd() + '/' + path + '../hotDINA/qmatrix.txt'
        self.new_student_params = new_student_params
        self.student_model_name = student_model_name
        self.hotDINA_skill_slurm_files = {}
        self.hotDINA_full_slurm_files = {}
        self.params_dict = {}
        self.path = path
        self.set_village_paths()
        self.set_data()
        self.set_slurm_files()
        self.set_params_dict()
        self.set_uniq_activities()
        self.set_new_student_params()
        exec(self.CONSTANTS['STUDENT_MODEL_INITIALISER'][student_model_name])
        print("StudentSimulator initialised (type: " + self.student_model_name + ')')

    def set_village_paths(self):
        village = self.CONSTANTS["VILLAGE"]
        file = self.path + "Data/village_" + village + "/village_" + village + "_step_transac.txt"
        self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"].append(file)
    
    def set_data(self):
        self.cta_df = read_cta_table(self.path + self.CONSTANTS["PATH_TO_CTA"])
        self.kc_list = self.cta_df.columns.tolist()
        
        if self.student_model_name == "ActivityBKT":
            self.activity_df = pd.read_pickle(self.path + self.CONSTANTS['PATH_TO_ACTIVITY_TABLE'])
            
            if self.CONSTANTS['OBSERVATIONS'] != 'all':
                num_obs = int(self.CONSTANTS['OBSERVATIONS'])
                self.activity_df = self.activity_df[:num_obs]

            self.set_uniq_activities()
            self.num_activities = len(self.uniq_activities)
            self.uniq_student_ids = get_col_vals_from_df(self.activity_df, "Unique_Child_ID_1", unique=True)
            self.uniq_student_ids.append("new_student")
        else:
            self.uniq_student_ids, self.student_id_to_village_map = get_uniq_transac_student_ids(self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"], [self.CONSTANTS["VILLAGE"]])
            self.uniq_student_ids.append("new_student")
            self.student_id_to_village_map['new_student'] = [self.CONSTANTS['VILLAGE']]
        
        self.kc_list_spaceless = remove_spaces(self.kc_list.copy())
    
    def set_slurm_files(self):
        self.set_hotDINA_skill_slurm_files()
        self.set_hotDINA_full_slurm_files()
    
    def set_hotDINA_skill_slurm_files(self):
        self.hotDINA_skill_slurm_files['130'] = "10301619"
    
    def set_hotDINA_full_slurm_files(self):
        # Remove this later, this slurm file is correct fit only for hotDINA_skill model, not hotDINA_full
        self.hotDINA_full_slurm_files['130'] = "10301619"

    def set_params_dict(self):
        village = self.CONSTANTS['VILLAGE']

        if self.student_model_name == "ItemBKT":
            self.params_dict    = get_bkt_params(self.kc_list_spaceless, 
                                                self.uniq_student_ids, 
                                                self.student_id_to_village_map, 
                                                [village], 
                                                self.CONSTANTS["SUBSCRIPT"],
                                                path=self.path) 

        elif self.student_model_name == 'ActivityBKT':

            n = len(self.uniq_student_ids)
            num_activities = self.num_activities
            u = len(self.kc_list)
            def_knew_skill      = 0.1 * np.ones((n, u))
            def_guess_skill     = 0.25 * np.ones((n, u))
            def_slip_skill      = 0.1 * np.ones((n, u))
            def_learn_skill     = 0.3 * np.ones((n, u))
            def_forget_skill    = np.zeros((n, u))
            def_knew_activity      = 0.1 * np.ones((n, num_activities))
            def_guess_activity     = 0.25 * np.ones((n, num_activities))
            def_slip_activity      = 0.1 * np.ones((n, num_activities))
            def_learn_activity     = 0.3 * np.ones((n, num_activities))
            def_forget_activity    = np.zeros((n, num_activities))

            self.params_dict = {
                'know_act'          :   def_knew_activity,
                'learn_act'         :   def_learn_activity,
                'guess_act'         :   def_guess_activity,
                'slip_act'          :   def_slip_activity,
                'forget_act'        :   def_forget_activity,
                'know'              :   def_knew_skill,
                'learn'             :   def_learn_skill,
                'guess'             :   def_guess_skill,
                'slip'              :   def_slip_skill,
                'forget'            :   def_forget_skill    
            }
            self.activity_learning_progress, self.act_student_id_to_number_map = init_act_student_id_to_number_map(n, u, self.uniq_student_ids, {}, def_knew_skill)
            
        else:
            path = os.getcwd() + '/' + self.path + "slurm_outputs"
            if self.student_model_name == "hotDINA_skill":
                self.params_dict = slurm_output_params(path, village, self.hotDINA_skill_slurm_files[village])
            elif self.student_model_name == 'hotDINA_full':
                self.params_dict = slurm_output_params(path, village, self.hotDINA_full_slurm_files[village])
            # student proficiency for 'new_student'
            self.params_dict['theta'].append(-np.log(9)/1.7)
        
    def set_uniq_activities(self):
        kc_to_tutorID_dict = init_kc_to_tutorID_dict(self.kc_list)
        self.cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, self.kc_list, self.cta_df)
        self.underscore_to_colon_tutor_id_dict = get_underscore_to_colon_tutor_id_dict(self.cta_tutor_ids)
        self.uniq_activities = get_uniq_activities(self.cta_tutor_ids, self.underscore_to_colon_tutor_id_dict)
    
    def set_new_student_params(self):

        if self.new_student_params == None:
            return
        
        student_model_name = self.student_model_name
        student_num = self.uniq_student_ids.index(self.new_student_params) 

        if student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
            self.params_dict['theta'][-1] = self.params_dict['theta'][student_num]


    def reset(self):
        student_model = self.student_model
        if self.student_model_name == 'ActivityBKT':
            student_model.know              = self.checkpoint_know.copy()
            student_model.know_act          = self.checkpoint_know_act.copy()
            student_model.learning_progress = self.checkpoint_learning_progress.copy()

        elif self.student_model_name == 'hotDINA_skill' or self.student_model_name == 'hotDINA_full':
            student_model.alpha     = self.checkpoint_know.copy()
            student_model.avg_knows = self.checkpoint_avg_knows.copy()
        
    def checkpoint(self):
        student_model = self.student_model
        
        if self.student_model_name == 'ActivityBKT':
            self.checkpoint_know                = student_model.know.copy()
            self.checkpoint_know_act            = student_model.know_act.copy()
            self.checkpoint_learning_progress   = student_model.learning_progress.copy()
        
        elif self.student_model_name == 'hotDINA_skill' or self.student_model_name == 'hotDINA_full':
            self.checkpoint_know                = student_model.alpha.copy()
            self.checkpoint_avg_knows           = student_model.avg_knows.copy()
        
    def update_on_log_data(self, data_dict, train_split=1.0, train_students=None, bayesian_update=True, plot=True):
        """
            Description: Takes a train_split (float) and train_students (list). If train_size is 1, trains on all rows of activity table, else trains on (1 - train_size)*rows of activity_df.
                        Update here means the BKT updates based on student performance (binary for ItemBKT, non binary for activity BKT since it takes in %correct as observation)
                        Only does ItemBKT for now. Can be changed to ActivityBKT later if need arises.
        """
        if train_students == None:
            train_students = self.uniq_student_ids.copy()
        
        if self.student_model_name == "ItemBKT":
            self.item_bkt_update(train_split, train_students)
        
        elif self.student_model_name == "ActivityBKT":
            self.activity_bkt_update(data_dict)
        
        elif self.student_model_name == 'hotDINA_skill':
            self.hotDINA_skill_update(data_dict, bayesian_update, plot, train_students=train_students)
        
        elif self.student_model_name == 'hotDINA_full':
            self.hotDINA_full_update(data_dict, train_students)

    def item_bkt_update(self, train_split, train_students):
        
        for student_id in tqdm(train_students, desc="student model update"):
            student_num = None
            if student_id not in self.uniq_student_ids:
                print("'", student_id, "' not present in student_id list. Assuming this is student 0 in village", self.CONSTANTS["VILLAGE"], "...")
                student_num = 0
            else:
                student_num = self.uniq_student_ids.index(student_id)
            
            if self.CONSTANTS["PRINT_PARAMS_FOR_STUDENT"] == True and self.student_model_name == 'ItemBKT':
                print("BKT PARAMS FOR STUDENT_ID: ", student_id)
                print("KNOW: ", self.params_dict['know'][student_num])
                print("LEARN: ", self.params_dict['learn'][student_num])
                print("GUESS: ", self.params_dict['guess'][student_num])
                print("SLIP: ", self.params_dict['slip'][student_num])
                print()
            
            for student_village in self.student_id_to_village_map[student_id]:

                data_file = "Data/village_" + str(student_village) + "/village_" + str(student_village) + "_step_transac.txt"
                student_ids, student_nums, skill_names, skill_nums, corrects, test_idx = extract_step_transac(data_file, self.uniq_student_ids, self.kc_list_spaceless, student_id, train_split)  

                if test_idx == None:
                    test_idx = len(student_ids)
                    self.student_model.update(student_nums, skill_nums, corrects)  
            
                else:
                    self.student_model.update(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])  
                    # full_resp_rmse, blame_worst_rmse = item_bkt.predict_percent_correct(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])
                    # total_blame_worst_rmse += blame_worst_rmse
                    # total_full_resp_rmse += full_resp_rmse
                print("Updated student_id:", pd.unique(np.array(student_ids)).tolist(), "with", str(test_idx), "rows")

    def activity_bkt_update(self, data_dict):
        activity_observations   = data_dict['activity_observations']
        student_nums            = data_dict['student_nums']
        activities              = data_dict['activity_names']                
        self.student_model.update(activity_observations, student_nums, activities)

    def hotDINA_skill_update(self, data_dict, bayesian_update=True, plot=True, train_students=None):
        if self.CONSTANTS["PRINT_PARAMS_FOR_STUDENT"] == True and self.student_model_name == 'hotDINA_skill':
            print("theta: ", self.params_dict['theta'])
            print('a: ', self.params_dict['a'])
            print('b: ', self.params_dict['b'])
            print("learn: ", self.params_dict['learn'])
            print("guess: ", self.params_dict['g'])
            print("ss: ", self.params_dict['ss'])
            print()

        observations = data_dict['y']
        items = data_dict['items']
        users = data_dict['users']
        self.student_model.update(observations, items, users, bayesian_update, plot, train_students)
    
    def hotDINA_full_update(self, data_dict, train_students):
        observations = data_dict['y']
        items = data_dict['items']
        users = data_dict['users']
        self.student_model.update(observations, items, users, train_students)

def check_ItemBKT(village, observations, student_simulator, train_ratio=0.75):
    path_to_step_transac_data   = 'Data/village_' + village + '/village_' + village + '_step_transac.txt'
    path_to_transac_table_data  = 'Data/village_' + village + '/village_' + village + '_KCSubtests.txt'
    uniq_student_ids            = student_simulator.uniq_student_ids
    kc_list_spaceless           = student_simulator.kc_list_spaceless
    uniq_student_ids, student_id_to_village_map = get_uniq_transac_student_ids([path_to_step_transac_data], [village])
    data_dict = extract_step_transac(path_to_step_transac_data, uniq_student_ids, kc_list_spaceless, observations=observations) 
    corrects            = data_dict['corrects']
    skill_nums          = data_dict['skill_nums']
    student_nums        = data_dict['student_nums']
    prob_rmse, sampled_rmse, majority_class_rmse, predicted_responses, predicted_p_corrects = student_simulator.student_model.get_rmse(student_nums, skill_nums, corrects, train_ratio=train_ratio)
    print("ItemBKT RMSE Value:", prob_rmse)
    print("Majority Class RMSE:", majority_class_rmse)

# FIXME 
def check_ActivityBKT(village, observations, student_simulator, student_id=None, train_size=1.0):
    activity_df = student_simulator.activity_df
    student_1_act_df = activity_df
    if student_id != None:
        student_1_act_df = activity_df[activity_df['Unique_Child_ID_1'] == student_id]
    data_dict = extract_activity_table("Data/Activity_table_v4.1_22Apr2020.pkl",
                                        student_simulator.uniq_activities, 
                                        student_simulator.kc_list)
    print(data_dict)
    # student_simulator.update_on_log_data(data_dict)
    # studs = []
    # for key in student_simulator.student_model.learning_progress:
    #     if len(student_simulator.student_model.learning_progress[key]) > 1:
    #         studs.append(key)
    # plot_learning(student_simulator.student_model.learning_progress, studs, 0, [], 'ppo')

def check_hotDINA_skill(village, observations, student_simulator, train_ratio=0.75):
    path_to_data_file = os.getcwd() + '/../hotDINA/pickles/data/data'+ village + '_' + observations +'.pickle'
    data_file = Path(path_to_data_file)
    if data_file.is_file() == False:
        # if data_file does not exist, get it
        os.chdir('../hotDINA')
        get_data_file_command = 'python get_data_for_village_n.py -v ' + village + ' -o ' + observations 
        os.system(get_data_file_command)
        os.chdir('../RoboTutor-Analysis')
    os.chdir('../hotDINA')
    with open(path_to_data_file, 'rb') as handle:
        data_dict = pickle.load(handle)
    os.chdir('../RoboTutor-Analysis')
    students = student_simulator.uniq_student_ids[:2]
    users = data_dict['users']
    items = data_dict['items']
    corrects = data_dict['y']
    rmse_val, majority_class_rmse = student_simulator.student_model.get_rmse(users, items, corrects, train_ratio=train_ratio)
    print("HotDINA_skill RMSE:", rmse_val)
    print("Majority class RMSE:", majority_class_rmse)

def check_hotDINA_full(village, observations, student_simulator, train_ratio=0.75):
    path_to_data_file = os.getcwd() + '/../hotDINA/pickles/data/data'+ village + '_' + observations +'.pickle'
    data_file = Path(path_to_data_file)
    if data_file.is_file() == False:
        # if data_file does not exist, get it
        os.chdir('../hotDINA')
        get_data_file_command = 'python get_data_for_village_n.py -v ' + village + ' -o ' + observations 
        os.system(get_data_file_command)
        os.chdir('../RoboTutor-Analysis')
    os.chdir('../hotDINA')
    with open(path_to_data_file, 'rb') as handle:
        data_dict = pickle.load(handle)
    os.chdir('../RoboTutor-Analysis')

    users = data_dict['users']
    items = data_dict['items']
    corrects = data_dict['y']
    majority_class_rmse, predicted_responses_rmse, predicted_p_corrects_rmse = student_simulator.student_model.get_rmse(users, items, corrects, train_ratio=train_ratio)
    print("Majority class RMSE for hotDINA_full:", majority_class_rmse)
    print("predicted_responses_rmse hotDINA_full:", predicted_responses_rmse)
    print("predicted_p_corrects_rmse hotDINA_full:", predicted_p_corrects_rmse)

def main(check_model, village, observations, train_ratio=0.75):
    student_simulator = StudentSimulator(village, observations, student_model_name=check_model)
    if check_model == 'ItemBKT':
        check_ItemBKT(village, observations, student_simulator, train_ratio)
    elif check_model == 'ActivityBKT':
        check_ActivityBKT(village, observations, student_simulator)
    elif check_model == 'hotDINA_skill':
        check_hotDINA_skill(village, observations, student_simulator, train_ratio)
    elif check_model == 'hotDINA_full':
        check_hotDINA_full(village, observations, student_simulator, train_ratio)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", help="NUM_ENTRIES that have to be extracted from a given transactions table. Should be a number or 'all'. If inputted number > total records for the village, this will assume a value of 'all'", default="1000", required=False)
    parser.add_argument("-v", "--village_num", help="village_num whose transactions data has to be extracted, should be between 114 and 141", default="130", required=False)
    args = parser.parse_args()
    village = args.village_num
    observations = args.observations
    main("hotDINA_full", village, observations, train_ratio=0.85)