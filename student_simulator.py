# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from student_models.item_bkt import ItemBKT
from student_models.activity_bkt import ActivityBKT
from helper import get_bkt_params, get_uniq_transac_student_ids, extract_step_transac, plot_learning, read_transac_table, get_kc_list_from_cta_table, remove_iter_suffix, init_kc_to_tutorID_dict, get_cta_tutor_ids,read_cta_table, get_col_vals_from_df, init_tutorID_to_kc_dict,init_underscore_to_colon_tutor_id_dict, init_item_learning_progress, extract_transac_table, init_skill_to_number_map, init_act_student_id_to_number_map, extract_activity_table, get_spaceless_kc_list
from tqdm import tqdm

class StudentSimulator():
    def __init__(self, villages, student_model_name="ItemBKT"):

        self.CONSTANTS = {
            "PATH_TO_CTA"                           : "Data/CTA.xlsx",
            "PATH_TO_ACTIVITY_TABLE"                : "Data/extracted_Activity_table_KCSubtest_sl2.xlsx",
            "VILLAGES"                              : villages,
            "SUBSCRIPT"                             : "student_specific", #"student_specific" or "village_specific" for bkt subscipting
            "PRINT_BKT_PARAMS_FOR_STUDENT"          : True,
            "PATH_TO_VILLAGE_STEP_TRANSAC_FILES"    :   []
        }

        for village in self.CONSTANTS["VILLAGES"]:
            file = "Data/village_" + village + "/village_" + village + "_step_transac.txt"
            self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"].append(file)

        self.cta_df = read_cta_table(self.CONSTANTS["PATH_TO_CTA"])
        self.kc_list = self.cta_df.columns.tolist()
        self.kc_list_spaceless = get_spaceless_kc_list(self.kc_list)
        self.uniq_student_ids, self.student_id_to_village_map = get_uniq_transac_student_ids(self.CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"], self.CONSTANTS["VILLAGES"])

        self.params_dict = get_bkt_params(self.kc_list_spaceless, 
                                        self.uniq_student_ids, 
                                        self.student_id_to_village_map, 
                                        self.CONSTANTS["VILLAGES"], 
                                        self.CONSTANTS["SUBSCRIPT"]) 
        
        # student_model_name can be "ItemBKT" or "ActivityBKT"
        self.student_model_name = student_model_name
        self.student_model = None

        if self.student_model_name == "ItemBKT":
            self.student_model = ItemBKT(self.params_dict, self.kc_list, self.uniq_student_ids)

        
    

    # def update_on_log_data(train_split, train_students=None):

    # """
    #     Description: Takes a train_split (float) and train_students (list). If train_size is 1, trains on all rows of activity table, else trains on (1 - train_size)*rows of activity_df.
    #                 Update here means the BKT updates based on student performance (binary for ItemBKT, non binary for activity BKT since it takes in %correct as observation)
    #                 Only does ItemBKT for now. Can be changed to ActivityBKT later if need arises.
    #     Returns:
    # """

    # if granularity == "item":

        

        
    #     item_bkt = ItemBKT(params_dict, kc_list, num_students, uniq_student_ids)

    #     if train_students == None:
    #         train_students = uniq_student_ids.copy()
        
    #     for student_id in tqdm(train_students, desc="student bkt update"):
    #         student_num = uniq_student_ids.index(student_id)
            
    #         if CONSTANTS["PRINT_BKT_PARAMS_FOR_STUDENT"] == True:
    #             print("BKT PARAMS FOR STUDENT_ID: ", student_id)
    #             print("KNOW: ", params_dict['know'][student_num])
    #             print("LEARN: ", params_dict['learn'][student_num])
    #             print("GUESS: ", params_dict['guess'][student_num])
    #             print("SLIP: ", params_dict['slip'][student_num])
    #             print()

    #         for student_village in student_id_to_village_map[student_id]:

    #             data_file = "Data/village_" + str(student_village) + "/village_" + str(student_village) + "_step_transac.txt"
    #             student_ids, student_nums, skill_names, skill_nums, corrects, test_idx = extract_step_transac(data_file, uniq_student_ids, kc_list_spaceless, student_id, train_split)  

    #             if test_idx == None:
    #                 test_idx = len(student_ids)
    #                 item_bkt.update(student_nums, skill_nums, corrects)  
            
    #             else:
    #                 item_bkt.update(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])  
    #                 # full_resp_rmse, blame_worst_rmse = item_bkt.predict_percent_correct(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])
    #                 # total_blame_worst_rmse += blame_worst_rmse
    #                 # total_full_resp_rmse += full_resp_rmse
    #             print("Updated student_id:", pd.unique(np.array(student_ids)).tolist(), "with", str(test_idx), "rows")

    #     return item_bkt

    # elif granularity == "activity":
    #     print("ACTIVITY BKT Student simulator")
    
    # else:
    #     print("Wrong value for 'granularity' at student_simulator.py")

    # return None

villages = ['114']
students = ['5A27001967', '5A27002160']
# item_bkt = update_on_obs(1.0, villages, train_students=students)
# plot_learning(item_bkt.learning_progress, students, 0, [], 'ppo')

student_simulator = StudentSimulator(villages)