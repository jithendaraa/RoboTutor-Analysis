# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import helper as hp
from helper import ItemBKT, ActivityBKT
from helper import plot_learning, read_transac_table, get_kc_list_from_cta_table, remove_iter_suffix, init_kc_to_tutorID_dict, get_cta_tutor_ids,read_cta_table, get_col_vals_from_df, init_tutorID_to_kc_dict,init_underscore_to_colon_tutor_id_dict, init_item_learning_progress, extract_transac_table, init_skill_to_number_map, init_act_student_id_to_number_map, extract_activity_table
from helper import sample_activity

def train_on_obs(test_size, train_students=None):

    """
        Description: Takes a test size (float) and train_students (list). If test_size is 1, trains on all rows of activity table, else trains on test_size*rows of activity_df.
                    Train here means the BKT updates based on student performance (binary for ItemBKT, non binary for activity BKT since it takes in %correct as observation)
                    Only does ActivityBKT for now. Can be changed to ItemBKT later if need arises.
        Returns:
            activity_bkt                    : The ActivityBKT Class which contains all details about updated P(Know), guess, slip and learn params for each student. 
                                                Also contains activity_learning_progress as an attribute. Useful to plot learning curves
            tutorID_to_kc_dict, 
            skill_to_number_map, 
            act_student_id_to_number_map    : 3 dicts which are initialised in this function and are very helpful to find relations between
                                            -> an activity and the kc's it exercises
                                            -> kc name to its index in kc_list
                                            -> "UNIQUE_CHILD_ID_1" to its index in uniq_activity_student_ids
    """
    
    # Paths
    # path_to_transac_table = "Data/transactions_village114.txt"
    path_to_cta_table = "Data/CTA.xlsx"
    path_to_activity_table = "Data/newActivityTable.csv"

    # Pandas data reading
    # transac_df = read_transac_table(path_to_transac_table, full_df=False)
    cta_df = read_cta_table(path_to_cta_table)
    activity_df = pd.read_csv(path_to_activity_table, dtype={"ActivityName": "str", "ProblemName": "str", "TotalProblemCount": str})

    # Variable initializations
    """
        kc_list                             : list of kc's from the CTA table
        u                                   : int. # kc's in the CTA or simply len(kc_list) 
        kc_to_tutorID_dict                  : dict mapping each kc to tutorIds that exercise the kc
        cta_tutor_ids                       : list containing all tutorIDs in the CTA table
        tutorID_to_kc_dict                  : dict that maps a tutorID to all kc's it exercises
        transac_tutor_ids                   : list containing all Tutor Ids as in the transactions table without "__it_" suffix
        uniq_transac_student_ids            : list unique student ids in the transactions table
        uniq_activity_student_ids           : list unique "UNIQUE_CHILD_ID1" in the activity table. Also contains a student_id = "new_student" which is not present in act. table
        n                                   : # unique student ids in the activity table
        student_id_to_number_map            : dict maps student_id to index of student_id in uniq_transac_student_ids
        underscore_to_colon_tutor_id_dict   : dict maps underscored tutorID to the colon tutorID [Note: all ":" become "_" but not vice versa]
        skill_to_number_map                 : dict kc name to its index in kc_list
        act_student_id_to_number_map        : dict maps "UNIQUE_CHILD_ID1" to index in uniq_activity_student_ids
        activity_observations               : list containing %corrects from each row of the activity table
        activity_learning_progress          : dict. activity_learning_progress["VPRQEF_101"] will give a 2d list and activity_learning_progress["VPRQEF_101"][i] is a 1d list containing the P(Know) for each skill after the ith opportunity
        activity_names                      : get the activity_names from each row of activity table, remove "__it_" suffix if any.
        Variables related to ItemBKT/transa tables commented out as of now
    """
    kc_list = get_kc_list_from_cta_table(cta_df)
    u = len(kc_list)
    kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
    cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)
    tutorID_to_kc_dict = init_tutorID_to_kc_dict(kc_to_tutorID_dict)
    # transac_tutor_ids = get_col_vals_from_df(transac_df, col_name="Level (Tutor)", unique=False)
    # transac_tutor_ids = remove_iter_suffix(transac_tutor_ids)
    # uniq_transac_student_ids = get_col_vals_from_df(transac_df, col_name="Anon Student Id", unique=True)
    # student_id_to_number_map = init_student_id_to_number_map(transac_df, type="transac")
    underscore_to_colon_tutor_id_dict = init_underscore_to_colon_tutor_id_dict(cta_tutor_ids)
    skill_to_number_map = init_skill_to_number_map(kc_list)
    act_student_id_to_number_map = {}
    activity_observations = []
    activity_learning_progress = {}
    activity_names = get_col_vals_from_df(activity_df, "ActivityName", unique=False)
    activity_names = remove_iter_suffix(activity_names)
    uniq_activity_student_ids = get_col_vals_from_df(activity_df, "Unique_Child_ID_1", unique=True)
    uniq_activity_student_ids.append("new_student")
    n = len(uniq_activity_student_ids)

    # Initial values of BKT params for activity. def stands for default. Ideally should be obtained after fitting BKT model on DataShop which will be done soon.
    def_activity_knew = 0.1 * np.ones((n, u))
    def_activity_guess = 0.25 * np.ones((n, u))
    def_activity_slip = 0.1 * np.ones((n, u))
    def_activity_learn = 0.3 * np.ones((n, u))
    def_activity_forget = np.zeros((n, u))

    activity_learning_progress, act_student_id_to_number_map = init_act_student_id_to_number_map(n, u, uniq_activity_student_ids, act_student_id_to_number_map, def_activity_knew)
    
    # student_id_counts[i] holds the #rows in which student with "UNIQUE_CHILD_ID_1" = uniq_activity_student_ids[i] occurs in the activity matrix. A count for each student.
    student_id_counts = []
    for student_id in uniq_activity_student_ids:
        count = activity_df[activity_df["Unique_Child_ID_1"] == student_id].count()[0]
        student_id_counts.append(count)
    
    # Init RMSE values by both responsibility methods. 
    total_full_resp_rmse = 0.0
    total_blame_worst_rmse = 0.0

    # Init ActivityBKT Class
    activity_bkt = ActivityBKT(def_activity_knew, def_activity_guess, def_activity_slip, def_activity_learn, kc_list, uniq_activity_student_ids, def_activity_forget, n, u, activity_learning_progress)

    for i in range(len(uniq_activity_student_ids)):
        
        # Train and do the BKT update only for students in train_students list; else skip
        if train_students != None:
            if uniq_activity_student_ids[i] not in train_students:
                continue
        
        # Get "UNIQUE_CHILD_ID_1" and counts of this student.
        student_id = uniq_activity_student_ids[i]
        count = student_id_counts[i]

        # test_df index for this student starts from test_idx
        if test_size != 1.0:
            test_idx = activity_df[activity_df["Unique_Child_ID_1"] == student_id].iloc[[math.floor(test_size*count)]].index.item()
            num_entries = test_idx
            train_activity_df = activity_df[:num_entries]
            test_activity_df = activity_df[num_entries:]
            train_activity_df = train_activity_df[train_activity_df["Unique_Child_ID_1"] == student_id]
            test_activity_df = test_activity_df[test_activity_df["Unique_Child_ID_1"] == student_id]
            train_rows = train_activity_df.shape[0]
            test_rows = test_activity_df.shape[0]

        else:    
            train_activity_df = activity_df[activity_df["Unique_Child_ID_1"] == student_id]
            train_rows = train_activity_df.shape[0]
            test_activity_df = None
            test_rows = 0

        n = len(uniq_activity_student_ids)
        
        # Get student observation data from activity table; like %correct, student_id, skills involved for that opportunity etc
        activity_observations, student_ids, activity_skills, num_corrects, num_attempts = extract_activity_table(train_activity_df, act_student_id_to_number_map, tutorID_to_kc_dict, skill_to_number_map, underscore_to_colon_tutor_id_dict)
        # Perform BKT update based on these row observations/opportunities
        activity_learning_progress = activity_bkt.update(activity_observations, student_ids, activity_skills)
        print("Done updating student_id:", student_id, " based on his/her responses! TRAIN_SET: ", train_rows, "TEST_SET: ", test_rows)

        # If test_size != 1.0, there will be a test_activity_df that is not empty; and we test on this data to find the RMSE's of the two types of predictions (full responsibility and blame-weakest)
        if test_activity_df is not None:
            actual_observations, student_ids, activity_skills, num_corrects, num_attempts = extract_activity_table(test_activity_df, act_student_id_to_number_map, tutorID_to_kc_dict, skill_to_number_map, underscore_to_colon_tutor_id_dict)
            full_resp_rmse, blame_worst_rmse = activity_bkt.predict_percent_correct(student_ids, activity_skills, actual_observations, num_corrects, num_attempts)
            total_blame_worst_rmse += blame_worst_rmse
            total_full_resp_rmse += full_resp_rmse

    if total_blame_worst_rmse + total_full_resp_rmse != 0.0:
        print(total_full_resp_rmse)
        print(total_blame_worst_rmse)
    
    return activity_bkt, tutorID_to_kc_dict, skill_to_number_map, act_student_id_to_number_map
