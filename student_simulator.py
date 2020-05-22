# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import helper as hp
from helper import ItemBKT, ActivityBKT
from helper import plot_learning, read_transac_table, get_kc_list_from_cta_table, remove_iter_suffix, init_kc_to_tutorID_dict, get_cta_tutor_ids,read_cta_table, get_col_vals_from_df, init_tutorID_to_kc_dict,init_underscore_to_colon_tutor_id_dict, init_item_learning_progress, init_student_id_to_number_map, extract_transac_table, init_skill_to_number_map, init_act_student_id_to_number_map, extract_activity_table

# Paths
path_to_transac_table = "Data/transactions_village114.txt"
path_to_cta_table = "Data/CTA.xlsx"
path_to_activity_table = "Data/newActivityTable.csv"

# Pandas data reading
transac_df = read_transac_table(path_to_transac_table, full_df=False)
cta_df = read_cta_table(path_to_cta_table)
activity_df = pd.read_csv(path_to_activity_table)

# Variable initializations
"""
    kc_list                             : list of kc's from the CTA table
    u                                   : # kc's in the CTA    
    kc_to_tutorID_dict                  : dict mapping each kc to tutorIds that exercise the kc
    cta_tutor_ids                       : list containing all tutorIDs in the CTA table
    tutorID_to_kc_dict                  : dict that maps a tutorID to all kc's it exercises
    transac_tutor_ids                   : list containing all Tutor Ids as in the transactions table without "__it_" suffix
    uniq_transac_student_ids            : unique student ids in the transactions table
    n                                   : # unique student ids in the transac table
    student_id_to_number_map            : dict maps student_id to index of student_id in uniq_transac_student_ids
    underscore_to_colon_tutor_id_dict   : dict maps underscored tutorID to the colon tutorID [Note: all ":" become "_" but not vice versa]

"""
kc_list = get_kc_list_from_cta_table(cta_df)
u = len(kc_list)
kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)
tutorID_to_kc_dict = init_tutorID_to_kc_dict(kc_to_tutorID_dict)
transac_tutor_ids = get_col_vals_from_df(transac_df, col_name="Level (Tutor)", unique=False)
transac_tutor_ids = remove_iter_suffix(transac_tutor_ids)
uniq_transac_student_ids = get_col_vals_from_df(transac_df, col_name="Anon Student Id", unique=True)
n = len(uniq_transac_student_ids)
student_id_to_number_map = init_student_id_to_number_map(transac_df)
underscore_to_colon_tutor_id_dict = init_underscore_to_colon_tutor_id_dict(cta_tutor_ids)
skill_to_number_map = init_skill_to_number_map(kc_list)
item_learning_progress = {}
act_student_id_to_number_map = {}
activity_observations = []
activity_learning_progress = {}

for ids in transac_tutor_ids:
    if ids not in cta_tutor_ids:
        print("ID MISSING IN CTA TABLE ", ids)


# Initial values for itemBKT variables/parameters. def stands for default
def_item_knew = 0.08 * np.ones((n, u))
def_item_guess = 0.22 * np.ones((n, u))
def_item_slip = 0.08 * np.ones((n, u))
def_item_learn = 0.2 * np.ones((n, u))
def_item_forget = np.zeros((n, u))

item_learning_progress = init_item_learning_progress(n, u, uniq_transac_student_ids)

# learning_progress[i][j][k] gives P(Knew) of student i skill j at timestep or opportunity k
item_observations = get_col_vals_from_df(transac_df, "Outcome", unique=False)
item_student_ids = get_col_vals_from_df(transac_df, "Anon Student Id", unique=False)
item_skills = [0] * len(item_observations)

item_observations, item_student_ids, item_skills, uniq_transac_student_ids = extract_transac_table(transac_df, 
                                                                                                    student_id_to_number_map, 
                                                                                                    kc_list, 
                                                                                                    skill_to_number_map, 
                                                                                                    underscore_to_colon_tutor_id_dict, 
                                                                                                    transac_tutor_ids, tutorID_to_kc_dict)

# ibkt = ItemBKT(def_item_knew, def_item_guess, def_item_slip, def_item_learn, kc_list, def_item_forget, n, u)
# item_learning_progress = ibkt.update(item_observations, item_student_ids, item_skills, item_learning_progress, uniq_transac_student_ids)
# plot_learning(item_learning_progress, kc_list, uniq_transac_student_ids)



# # Activity BKT

activity_names = get_col_vals_from_df(activity_df, "ActivityName", unique=False)
activity_names = remove_iter_suffix(activity_names)
uniq_activity_student_ids = get_col_vals_from_df(activity_df, "Unique_Child_ID_1", unique=True)
n = len(uniq_activity_student_ids)

# Initial values for activity BKT variables/parameters. def stands for default
def_activity_knew = 0.1 * np.ones((n, u))
def_activity_guess = 0.25 * np.ones((n, u))
def_activity_slip = 0.1 * np.ones((n, u))
def_activity_learn = 0.3 * np.ones((n, u))
def_activity_forget = np.zeros((n, u))
activity_bkt = ActivityBKT(def_activity_knew, def_activity_guess, def_activity_slip, def_activity_learn, kc_list, def_activity_forget, n, u)

student_id_counts = []
for student_id in uniq_activity_student_ids:
    count = activity_df[activity_df["Unique_Child_ID_1"] == student_id].count()[0]
    student_id_counts.append(count)

activity_learning_progress, act_student_id_to_number_map = init_act_student_id_to_number_map(n, u, uniq_activity_student_ids, act_student_id_to_number_map)

total_full_resp_rmse = 0.0
total_blame_worst_rmse = 0.0

for i in range(len(uniq_activity_student_ids)):
    student_id = uniq_activity_student_ids[i]
    count = student_id_counts[i]
    # Index of test_df for this student starts from test_idx
    test_size = 0.8
    test_idx = activity_df[activity_df["Unique_Child_ID_1"] == student_id].iloc[[math.floor(0.8*count)]].index.item()

    num_entries = test_idx
    print(num_entries)
    train_activity_df = activity_df[:num_entries]
    test_activity_df = activity_df[num_entries:]

    train_activity_df = train_activity_df[train_activity_df["Unique_Child_ID_1"] == student_id]
    test_activity_df = test_activity_df[test_activity_df["Unique_Child_ID_1"] == student_id]
    train_rows = train_activity_df.shape[0]
    test_rows = test_activity_df.shape[0]

    n = len(uniq_activity_student_ids)
    activity_observations, student_ids, activity_skills, num_corrects, num_attempts = extract_activity_table(train_activity_df, act_student_id_to_number_map, tutorID_to_kc_dict, skill_to_number_map, underscore_to_colon_tutor_id_dict)
    activity_learning_progress = activity_bkt.update(activity_observations, student_ids, activity_skills, activity_learning_progress, uniq_activity_student_ids)
    print("Done updating student_id: ", student_id, "TRAIN: ", train_rows, "TEST: ", test_rows)

    actual_observations, student_ids, activity_skills, num_corrects, num_attempts = extract_activity_table(test_activity_df, act_student_id_to_number_map, tutorID_to_kc_dict, skill_to_number_map, underscore_to_colon_tutor_id_dict)
    full_resp_rmse, blame_worst_rmse = activity_bkt.predict_percent_correct(student_ids, activity_skills, num_corrects, num_attempts, activity_learning_progress, uniq_activity_student_ids, actual_observations)
    
    total_blame_worst_rmse += blame_worst_rmse
    total_full_resp_rmse += full_resp_rmse

print(total_full_resp_rmse)
print(total_blame_worst_rmse)