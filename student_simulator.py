# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from reader import *
from helper import *

# from student_models.item_bkt import ItemBKT
# from student_models.activity_bkt import ActivityBKT
from student_models.hotDINA_skill import hotDINA_skill


class StudentSimulator():
    def __init__(self, village="130", observations="1000", student_model_name="ItemBKT", subscript="student_specific"):

        self.CONSTANTS = {
            "PATH_TO_CTA"                           : "Data/CTA.xlsx",
            "PATH_TO_ACTIVITY_TABLE"                : "Data/extracted_Activity_table_KCSubtest_sl2.xlsx",
            "VILLAGE"                               : village,
            "OBSERVATIONS"                          : observations,
            "SUBSCRIPT"                             : subscript, #"student_specific" or "village_specific" for BKT subscripting
            "PRINT_PARAMS_FOR_STUDENT"              : True,
            "PATH_TO_VILLAGE_STEP_TRANSAC_FILES"    : [],
            "STUDENT_MODEL_INITIALISER"             : {
                                                        # "ItemBKT"       : "self.student_model = ItemBKT(self.params_dict, self.kc_list, self.uniq_student_ids)",
                                                        # "ActivityBKT"   : "self.student_model = ActivityBKT(self.params_dict, self.kc_list, self.uniq_student_ids)",
                                                        "hotDINA_skill" : "self.student_model = hotDINA_skill(self.params_dict, path_to_Qmatrix)"
                                                    },
        }
        path_to_Qmatrix = os.getcwd() + '/../hotDINA/qmatrix.txt'

        self.student_model_name = student_model_name
        self.hotDINA_skill_slurm_files = {}
        self.hotDINA_full_slurm_files = {}
        self.params_dict = {}
        
        # Set village path constants, RT village data and params for the student_model
        self.set_village_paths()
        self.set_data()
        self.set_slurm_files()
        self.set_params_dict()
        exec(self.CONSTANTS['STUDENT_MODEL_INITIALISER'][student_model_name])

    def set_village_paths(self):
        village = self.CONSTANTS["VILLAGE"]
        file = "Data/village_" + village + "/village_" + village + "_step_transac.txt"
        self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"].append(file)
    
    def set_data(self):
        self.cta_df = read_cta_table(self.CONSTANTS["PATH_TO_CTA"])
        self.kc_list = self.cta_df.columns.tolist()
        self.kc_list_spaceless = remove_spaces(self.kc_list)
        self.uniq_student_ids, self.student_id_to_village_map = get_uniq_transac_student_ids(self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"], [self.CONSTANTS["VILLAGE"]])
        self.student_id_to_village_map['new_student'] = [int(self.CONSTANTS['VILLAGE'])]
    
    def set_slurm_files(self):
        self.set_hotDINA_skill_slurm_files()
        self.set_hotDINA_full_slurm_files()
    
    def set_hotDINA_skill_slurm_files(self):
        self.hotDINA_skill_slurm_files['130'] = "10301619"
    
    def set_hotDINA_full_slurm_files(self):
        pass

    def set_params_dict(self):
        if self.student_model_name == "ItemBKT":
            self.params_dict    = get_bkt_params(self.kc_list_spaceless, 
                                                self.uniq_student_ids, 
                                                self.student_id_to_village_map, 
                                                [self.CONSTANTS["VILLAGE"]], 
                                                self.CONSTANTS["SUBSCRIPT"]) 
        
        elif self.student_model_name == "hotDINA_skill":
            path = os.getcwd() + "/slurm_outputs"
            self.params_dict = slurm_output_params(path, village, self.hotDINA_skill_slurm_files[village])

        # self.student_model_name = student_model_name
        # self.student_model = exec(self.CONSTANTS['STUDENT_MODEL_INITIALISER'][self.student_model_name])       

    def update_on_log_data(self, train_split, data_dict, train_students=None, bayesian_update=True, plot=True):
        """
            Description: Takes a train_split (float) and train_students (list). If train_size is 1, trains on all rows of activity table, else trains on (1 - train_size)*rows of activity_df.
                        Update here means the BKT updates based on student performance (binary for ItemBKT, non binary for activity BKT since it takes in %correct as observation)
                        Only does ItemBKT for now. Can be changed to ActivityBKT later if need arises.
        """
        if train_students == None:
            train_students = self.uniq_student_ids.copy()
        
        if self.student_model_name == "ItemBKT":
            self.item_bkt_update(train_split, train_students)
        
        elif self.student_model_name == 'hotDINA_skill':
            self.hotDINA_skill_update(train_students, data_dict, bayesian_update, plot)

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
                student_ids, student_nums, skill_names, skill_nums, corrects, test_idx = extract_step_transac(data_file, 
                                                                                                                self.uniq_student_ids, 
                                                                                                                self.kc_list_spaceless, 
                                                                                                                student_id, 
                                                                                                                train_split)  

                if test_idx == None:
                    test_idx = len(student_ids)
                    self.student_model.update(student_nums, skill_nums, corrects)  
            
                else:
                    self.student_model.update(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])  
                    # full_resp_rmse, blame_worst_rmse = item_bkt.predict_percent_correct(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])
                    # total_blame_worst_rmse += blame_worst_rmse
                    # total_full_resp_rmse += full_resp_rmse
                print("Updated student_id:", pd.unique(np.array(student_ids)).tolist(), "with", str(test_idx), "rows")

    def hotDINA_skill_update(self, train_students, data_dict, bayesian_update=True, plot=True):

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
        self.student_model.update(observations, items, users, bayesian_update, plot)

if __name__ == "__main__":

    from pathlib import Path
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", help="NUM_ENTRIES that have to be extracted from a given transactions table. Should be a number or 'all'. If inputted number > total records for the village, this will assume a value of 'all'")
    parser.add_argument("-v", "--village_num", help="village_num whose transactions data has to be extracted, should be between 114 and 141")
    args = parser.parse_args()

    village = args.village_num
    observations = args.observations

    student_simulator = StudentSimulator(village, observations, student_model_name="hotDINA_skill")
    
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
    student_simulator.update_on_log_data(1.0, train_students=students, data_dict=data_dict, bayesian_update=False, plot=True)
    plt.show()
    # plot_learning(student_simulator.student_model.learning_progress, students, 0, [], 'ppo')