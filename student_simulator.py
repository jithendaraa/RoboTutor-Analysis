# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from helper import ItemBKT, get_difficulties
from helper import get_bkt_params, get_uniq_transac_student_ids, extract_step_transac, plot_learning, read_transac_table, get_kc_list_from_cta_table, remove_iter_suffix, init_kc_to_tutorID_dict, get_cta_tutor_ids,read_cta_table, get_col_vals_from_df, init_tutorID_to_kc_dict,init_underscore_to_colon_tutor_id_dict, init_item_learning_progress, extract_transac_table, init_skill_to_number_map, init_act_student_id_to_number_map, extract_activity_table, get_spaceless_kc_list
from tqdm import tqdm

# granularity can be "item" or "activity" based on item-level BKT or activity_level_BKT
def train_on_obs(train_split, train_students=None, granularity="item"):

    """
        Description: Takes a test size (float) and train_students (list). If test_size is 1, trains on all rows of activity table, else trains on test_size*rows of activity_df.
                    Train here means the BKT updates based on student performance (binary for ItemBKT, non binary for activity BKT since it takes in %correct as observation)
                    Only does ActivityBKT for now. Can be changed to ItemBKT later if need arises.
        Returns:
    """

    CONSTANTS = {
                    "PATH_TO_CTA"                   : "Data/CTA.xlsx",
                    "PATH_TO_ACTIVITY_TABLE"        : "Data/extracted_Activity_table_KCSubtest_sl2.xlsx",
                    "VILLAGES"                      : ['114'],
                    # "VILLAGES"                      : ['114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141'],
                    "SUBSCRIPT"                     : "student_specific", #"student_specific" or "village_specific" for bkt subscipting
                    "PRINT_BKT_PARAMS_FOR_STUDENT"  : True,
                }

    if granularity == "item":
        CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"] = []

        for village in CONSTANTS["VILLAGES"]:
            file = "Data/village_" + village + "/village_" + village + "_step_transac.txt"
            CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"].append(file)

        cta_df = read_cta_table(CONSTANTS["PATH_TO_CTA"])
        kc_list = cta_df.columns.tolist()
        kc_list_spaceless = get_spaceless_kc_list(kc_list)
        uniq_student_ids, student_id_to_village_map = get_uniq_transac_student_ids(CONSTANTS["PATH_TO_VILLAGE_STEP_TRANSAC_FILES"], CONSTANTS["VILLAGES"])

        num_skills = len(kc_list)
        num_students = len(uniq_student_ids)

        init_know, learn, slip, guess, forget = get_bkt_params(num_skills, kc_list_spaceless, uniq_student_ids, num_students, student_id_to_village_map, CONSTANTS["VILLAGES"], CONSTANTS["SUBSCRIPT"]) 
        item_bkt = ItemBKT(init_know, guess, slip, learn, kc_list, forget, num_students, num_skills, uniq_student_ids)

        # total_full_resp_rmse = 0.0
        # total_blame_worst_rmse = 0.0

        if train_students == None:
            train_students = uniq_student_ids.copy()
        
        for student_id in tqdm(train_students, desc="student bkt update"):
            student_num = uniq_student_ids.index(student_id)
            
            if CONSTANTS["PRINT_BKT_PARAMS_FOR_STUDENT"] == True:
                print("BKT PARAMS FOR STUDENT_ID: ", student_id)
                print("KNOW: ", init_know[student_num])
                print("LEARN: ", learn[student_num])
                print("GUESS: ", guess[student_num])
                print("SLIP: ", slip[student_num])
                print()

            for student_village in student_id_to_village_map[student_id]:

                data_file = "Data/village_" + str(student_village) + "/village_" + str(student_village) + "_step_transac.txt"
                student_ids, student_nums, skill_names, skill_nums, corrects, test_idx = extract_step_transac(data_file, uniq_student_ids, kc_list_spaceless, student_id, train_split)  

                if test_idx == None:
                    test_idx = len(student_ids)
                    item_bkt.update(student_nums, skill_nums, corrects)  
            
                else:
                    item_bkt.update(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])  
                    # full_resp_rmse, blame_worst_rmse = item_bkt.predict_percent_correct(student_nums[:test_idx], skill_nums[:test_idx], corrects[:test_idx])
                    # total_blame_worst_rmse += blame_worst_rmse
                    # total_full_resp_rmse += full_resp_rmse
                print("Updated student_id:", pd.unique(np.array(student_ids)).tolist(), "with", str(test_idx), "rows")

        return item_bkt
    elif granularity == "activity":
        print("ACTIVITY BKT Student simulator")
    else:
        print("Wrong Granularity")

    return None
    
students = ['5A27001967', '5A27002160']
item_bkt = train_on_obs(1.0, train_students=students)
# print(item_bkt.learning_progress['5A22000679'])
plot_learning(item_bkt.learning_progress, students, 0, [], 'ppo')