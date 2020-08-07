# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from reader import *
from helper import *

from student_models.item_bkt import ItemBKT
from student_models.activity_bkt import ActivityBKT


class StudentSimulator():
    def __init__(self, villages=["114"], student_model_name="ItemBKT", T=None):

        self.CONSTANTS = {
            "PATH_TO_CTA"                           : "Data/CTA.xlsx",
            "PATH_TO_ACTIVITY_TABLE"                : "Data/extracted_Activity_table_KCSubtest_sl2.xlsx",
            "VILLAGES"                              : villages,
            "SUBSCRIPT"                             : "student_specific", #"student_specific" or "village_specific" for bkt subscipting
            "PRINT_BKT_PARAMS_FOR_STUDENT"          : True,
            "PATH_TO_VILLAGE_STEP_TRANSAC_FILES"    :   [],
            "STUDENT_MODEL_INITIALISER"             : {
                                                            "ItemBKT"       : "self.student_model = ItemBKT(self.params_dict, self.kc_list, self.uniq_student_ids)",
                                                            "ActivityBKT"   : "self.student_model = ActivityBKT(self.params_dict, self.kc_list, self.uniq_student_ids)",
                                                            "hotDINA_skill" : "self.student_model = hotDINA_skill(self.params_dict, T)"
                                                        }
        }

        for village in self.CONSTANTS["VILLAGES"]:
            file = "Data/village_" + village + "/village_" + village + "_step_transac.txt"
            self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"].append(file)

        self.cta_df = read_cta_table(self.CONSTANTS["PATH_TO_CTA"])
        self.kc_list = self.cta_df.columns.tolist()
        self.kc_list_spaceless = remove_spaces(self.kc_list)
        self.uniq_student_ids, self.student_id_to_village_map = get_uniq_transac_student_ids(self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"], self.CONSTANTS["VILLAGES"])

        self.student_id_to_village_map['new_student'] = [int(self.CONSTANTS['VILLAGES'][0])]

        self.params_dict = get_bkt_params(self.kc_list_spaceless, 
                                        self.uniq_student_ids, 
                                        self.student_id_to_village_map, 
                                        self.CONSTANTS["VILLAGES"], 
                                        self.CONSTANTS["SUBSCRIPT"]) 
        
        self.student_model_name = student_model_name
        self.student_model = self.CONSTANTS['STUDENT_MODEL_INITIALISER'][self.student_model_name]
            

    def update_on_log_data(self, train_split, train_students=None):
        """
            Description: Takes a train_split (float) and train_students (list). If train_size is 1, trains on all rows of activity table, else trains on (1 - train_size)*rows of activity_df.
                        Update here means the BKT updates based on student performance (binary for ItemBKT, non binary for activity BKT since it takes in %correct as observation)
                        Only does ItemBKT for now. Can be changed to ActivityBKT later if need arises.
        """
        if train_students == None:
            train_students = self.uniq_student_ids.copy()
        
        for student_id in tqdm(train_students, desc="student model update"):
            
            student_num = None
            if student_id not in self.uniq_student_ids:
                print("'", student_id, "' not present in student_id list. Assuming this is student 0 in village", self.CONSTANTS["VILLAGES"][0], "...")
                student_num = 0
            else:
                student_num = self.uniq_student_ids.index(student_id)
            
            if self.CONSTANTS["PRINT_BKT_PARAMS_FOR_STUDENT"] == True:
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


if __name__ == "__main__":
    
    villages = ['114']
    students = ['5A27001967', '5A27002160']
    student_simulator = StudentSimulator(villages)

    print(student_simulator.student_model)
    # student_simulator.update_on_log_data(1.0, train_students=students)
    # plot_learning(student_simulator.student_model.learning_progress, students, 0, [], 'ppo')

    # villages = ["130"]


