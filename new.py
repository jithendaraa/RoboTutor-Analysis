# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from helper import ItemBKT, ActivityBKT
from helper import plot_learning, read_transac_table, get_kc_list_from_cta_table, remove_iter_suffix, init_kc_to_tutorID_dict, get_cta_tutor_ids,read_cta_table, get_col_vals_from_df, init_tutorID_to_kc_dict,init_underscore_to_colon_tutor_id_dict, init_item_learning_progress, extract_transac_table, init_skill_to_number_map, init_act_student_id_to_number_map, extract_activity_table, get_difficulties

path_to_cta_table = "Data/CTA.xlsx"
path_to_activity_table = "Data/extracted_Activity_table_KCSubtest_sl2.xlsx"
cta_df = read_cta_table(path_to_cta_table)
activity_df = pd.read_excel(path_to_activity_table)
kc_list = get_kc_list_from_cta_table(cta_df)

kc_vals = pd.unique(activity_df['KC (Subtest)'].values.ravel()).tolist()
for val in kc_vals:
    if val not in kc_list:
        print(val)
        kc_list.append(val)

u = len(kc_list)
kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)

tutorID_to_kc_dict = init_tutorID_to_kc_dict(kc_to_tutorID_dict)
underscore_to_colon_tutor_id_dict = init_underscore_to_colon_tutor_id_dict(cta_tutor_ids)
skill_to_number_map = init_skill_to_number_map(kc_list)
act_student_id_to_number_map = {}
activity_observations = []
activity_learning_progress = {}

activity_names = activity_df["ActivityName"].values.tolist()

uniq_activities = []
for tutor_id in cta_tutor_ids:
    if underscore_to_colon_tutor_id_dict[tutor_id] not in uniq_activities:
        uniq_activities.append(underscore_to_colon_tutor_id_dict[tutor_id])

uniq_activity_student_ids = get_col_vals_from_df(activity_df, "Unique_Child_ID_1", unique=True)
uniq_activity_student_ids.append("new_student")

n = len(uniq_activity_student_ids)
num_activities = len(cta_tutor_ids)

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

activity_learning_progress, act_student_id_to_number_map = init_act_student_id_to_number_map(n, u, uniq_activity_student_ids, act_student_id_to_number_map, def_knew_skill)
difficulty = get_difficulties(u, tutorID_to_kc_dict, kc_list)

activity_bkt = ActivityBKT(def_knew_skill, 
                            def_guess_skill, 
                            def_slip_skill, 
                            def_learn_skill, 
                            kc_list, 
                            uniq_activity_student_ids, 
                            def_forget_skill, 
                            n, 
                            u, 
                            activity_learning_progress,
                            def_knew_activity,
                            def_guess_activity,
                            def_slip_activity,
                            def_learn_activity,
                            def_forget_activity,
                            difficulty)

activity_bkt.set_coefficients(uniq_activities)

math_activity_df = activity_df[activity_df["Matrix_ActivityName"] == 'math']
student_1_act_df = activity_df[activity_df['Unique_Child_ID_1'] == 'PWCRKF_379']

activity_observations, student_nums, activity_skills, num_corrects, num_attempts, activities = extract_activity_table(student_1_act_df, act_student_id_to_number_map, kc_list)

activity_bkt.update_by_activity(activity_observations, student_nums, activities, cta_tutor_ids, underscore_to_colon_tutor_id_dict)

studs = []

for key in activity_bkt.activity_learning_progress:
    if len(activity_bkt.activity_learning_progress[key]) > 1:
        studs.append(key)

plot_learning(activity_learning_progress, studs, 0, [], 'ppo')