"""
    Contains all helper functions that are used to read and extract RoboTutor data 
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tqdm import tnrange, tqdm_notebook, tqdm
import pickle


# Read functions
def read_transac_table(path_to_transac_table, full_df=False):
    """
        Reads and returns transac table as a df after dropping some columns that aren't required for our use
    """
    transac_df = pd.read_csv(path_to_transac_table, sep='\t')
    # Make Anon Student Id uniform data type 'str'
    transac_df = transac_df.astype({"Anon Student Id": str})
    
    if full_df == False:
        drop_cols = ["Row", "Sample Name", "Session Id","Time","Problem Start Time","Time Zone", "Duration (sec)", "Student Response Type", "Student Response Subtype", "Tutor Response Type", "Tutor Response Subtype", "Selection", "Action", "Feedback Text", "Feedback Classification", "Help Level", "Total Num Hints", "School", "Class", "CF (File)", "CF (Hiatus sec)", "CF (Original DS Export File)","CF (Unix Epoch)","CF (Village)","Event Type","CF (Student Used Scaffold)","CF (Robotutor Mode)","KC Category (Single-KC)","KC Category (Unique-step)","CF (Child Id)","CF (Date)","CF (Placement Test Flag)","CF (Week)","Transaction Id","Problem View","KC (Single-KC)","KC (Unique-step)", "CF (Activity Finished)", "CF (Activity Started)", "CF (Attempt Number)","CF (Duration sec)", "CF (Expected Answer)", "CF (Matrix)", "CF (Matrix Level)", "CF (Matrix Order)", "CF (Original Order)","CF (Outcome Numeric)", "CF (Placement Test User)", "CF (Problem Number)", "CF (Session Sequence)", "CF (Student Chose Repeat)", "CF (Total Activity Problems)", "CF (Tutor Sequence Session)", "CF (Tutor Sequence User)","Input", "Is Last Attempt", "Attempt At Step", "Step Name"]
        transac_df = transac_df.drop(columns=drop_cols)

    return transac_df

def read_cta_table(path_to_cta_table):
    """
        Reads and returns the CTA table as a df
    """
    cta_df = pd.read_excel(path_to_cta_table).astype({'Quantifying': str})
    return cta_df

def read_data(path=""):
    """
        Reads and returns some useful data from CTA Table and activity_table.
    """
    cta_df = read_cta_table(path + "Data/CTA.xlsx")
    kc_list = get_kc_list_from_cta_table(cta_df)
    num_skills = len(kc_list)
    kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
    cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)
    tutorID_to_kc_dict = get_tutorID_to_kc_dict(kc_to_tutorID_dict)
    uniq_skill_groups, skill_group_to_activity_map = get_skill_groups_info(tutorID_to_kc_dict, kc_list)
    print("DATA READING DONE.....")
    
    return kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map

def read_activity_matrix(PATH_TO_ACTIVITY_DIFFICULTY, LITERACY_SHEET_NAME, MATH_SHEET_NAME, STORIES_SHEET_NAME):
    """
        Takes paths and sheet names as params and returns the 3 activity matrices
    """
    xls = pd.ExcelFile(PATH_TO_ACTIVITY_DIFFICULTY)

    # Difficulty with levels as rows
    literacy_df = pd.read_excel(xls, LITERACY_SHEET_NAME)[1:]
    math_df = pd.read_excel(xls, MATH_SHEET_NAME)
    stories_df = pd.read_excel(xls, STORIES_SHEET_NAME)

    literacy_matrix = literacy_df.values.tolist()
    math_matrix = math_df.values.tolist()   
    stories_matrix = stories_df.values.tolist()
    stories_matrix.insert(0, stories_df.columns.tolist())

    return literacy_matrix, math_matrix, stories_matrix


# Extract functions
def extract_transac_table(transac_df, student_id_to_number_map, kc_list, skill_to_number_map, underscore_to_colon_tutor_id_dict, transac_tutor_ids, tutorID_to_kc_dict):
    """
        Gets all details from transactions table that are required to do the ItemBKT update and return these details
        observations, student_ids, skills involved with these opportuniutes and uniq_transac_student_ids. All these 4 variables are lists and ith row gives details for ith opportunity
    """
    item_observations = get_col_vals_from_df(transac_df, "Outcome", unique=False)
    item_student_ids = get_col_vals_from_df(transac_df, "Anon Student Id", unique=False)
    uniq_transac_student_ids = get_col_vals_from_df(transac_df, "Anon Student Id", unique=True)
    item_skills = [0] * len(item_observations)

    for i in range(len(item_student_ids)):
        item_student_ids[i] = student_id_to_number_map[item_student_ids[i]]
    
    for i in range(len(item_skills)):
        underscore_tutor_id = transac_tutor_ids[i]
        colon_tutor_id = underscore_to_colon_tutor_id_dict[underscore_tutor_id]
        item_skills[i] = tutorID_to_kc_dict[colon_tutor_id]
        res = []
        for row in item_skills[i]:
            res.append(skill_to_number_map[row])
        item_skills[i] = res

    student_nums = []
    for student_id in item_student_ids:
        student_nums.append(uniq_transac_student_ids.index(student_id))

    data_dict = {
        'observations'              : item_observations,
        'student_ids'               : item_student_ids,
        'student_nums'              : student_nums,
        'item_skills'               : item_skills,
        'uniq_transac_student_ids'  : uniq_transac_student_ids
    }

    return data_dict

def extract_step_transac(path_to_data, uniq_student_ids, kc_list_spaceless, student_id=None, train_split=1.0, observations='all'):

    student_ids = []
    student_nums = []
    skill_names = []
    skill_nums = []
    corrects = []

    df = pd.read_csv(path_to_data, delimiter='\t', header=None).astype({1: str})
    
    if observations != 'all':
        df = df[:int(observations)]
    
    corrects = df[0].values.tolist()
    student_ids = df[1].values.tolist()
    skill_names = df[3].values.tolist()
    skill_nums = df[3].values.tolist()
       
    for stud_id in student_ids:
        student_nums.append(uniq_student_ids.index(stud_id))

    for i in range(len(skill_names)):
        skill_names[i] = skill_names[i].split("~")
        skill_nums[i] = skill_nums[i].split("~")

    for i in range(len(skill_nums)):
        row = skill_nums[i]
        for j in range(len(row)):
            val = row[j]
            skill_nums[i][j] = kc_list_spaceless.index(val)  

    for i in range(len(corrects)):
        if corrects[i] == 2:
            corrects[i] = 0
    
    if student_id != None:
        for i in range(len(corrects)):
            while i<len(student_ids) and student_ids[i] != student_id:
                student_ids.pop(i)
                student_nums.pop(i)
                corrects.pop(i)
                skill_names.pop(i)
                skill_nums.pop(i)

    num_entries = len(student_ids)
    test_idx = None
    if train_split != 1.0:
        test_idx = math.floor(train_split * num_entries)

    data_dict = {
        'student_ids'   : student_ids,
        'student_nums'  : student_nums,
        'skill_names'   : skill_names,
        'skill_nums'    : skill_nums,
        'corrects'      : corrects,
        'test_idx'      : test_idx
    }
    
    return data_dict

def extract_activity_table(path_to_activity_table, uniq_student_ids, kc_list, num_obs='1000', student_id=None):
    """
        Reads actiivty table, gets necessary data from it and gets all details that are necessary to do the activityBKT update.
        Returns observations, student_ids, skills involved with each of these attempts, num_corrects and num_attempts
        ith row of each of these lists give info about the ith opportunity or corresponds to ith row of activity_df
    """
    activity_df = pd.read_pickle(path_to_activity_table)
    print(activity_df)
    if student_id != None:
        activity_df = activity_df[activity_df["Unique_Child_ID_1"] == student_id]
    if num_obs != 'all':
        activity_df = activity_df[:int(num_obs)]

    activity_observations = get_col_vals_from_df(activity_df, "%correct", unique=False)
    student_ids = get_col_vals_from_df(activity_df, "Unique_Child_ID_1", unique=False)
    num_corrects = get_col_vals_from_df(activity_df, "total_correct_attempts", unique=False)
    num_attempts = get_col_vals_from_df(activity_df, "#net_attempts", unique=False)
    activity_names = get_col_vals_from_df(activity_df, "ActivityName", unique=False)
    kc_subtests = get_col_vals_from_df(activity_df, "KC (Subtest)", unique=False)

    student_nums = []
    for student_id in student_ids:
        student_nums.append(uniq_student_ids.index(student_id))
    
    activity_skills = []
    for kc_subtest in kc_subtests:
        if kc_subtest == 'Listening Comp':
            kc_subtest = 'Listening Comprehension'
        elif kc_subtest == 'Number I.D.':
            kc_subtest = 'Number. I.D'
        activity_skills.append(kc_list.index(kc_subtest)) 

    data_dict = {
        # 'activity_observations' : activity_observations,
        'student_nums'          : student_nums,
        'activity_skills'       : activity_skills,
        'num_corrects'          : num_corrects,
        'num_attempts'          : num_attempts,
        'activity_names'        : activity_names
    }

    return data_dict

# Get functions
def get_uniq_transac_student_ids(PATH_TO_VILLAGE_STEP_TRANSAC_FILES, villages):
    uniq_student_ids = []
    village_to_student_id_map = {}
    student_id_to_village_map = {}

    for i in range(len(PATH_TO_VILLAGE_STEP_TRANSAC_FILES)):

        path_to_file = PATH_TO_VILLAGE_STEP_TRANSAC_FILES[i]
        village = villages[i]

        df = pd.read_csv(path_to_file, delimiter='\t', header=None, dtype={1: str})

        student_ids = pd.unique(df[1].values.ravel()).tolist()
        village_to_student_id_map[village] = student_ids
        uniq_student_ids = uniq_student_ids + student_ids
    
    for village in village_to_student_id_map:
        value = village_to_student_id_map[village]
        
        for student_id in value:
            student_id_to_village_map[student_id] = []

    for village in village_to_student_id_map:
        value = village_to_student_id_map[village]
        
        for student_id in value:
            student_id_to_village_map[student_id].append(village)

    return uniq_student_ids, student_id_to_village_map

def get_tutorID_to_kc_dict(kc_to_tutorID_dict):
    """
        Init and populate tutorID_to_kc_dict
    """
    tutorID_to_kc_dict = {}
    for key in kc_to_tutorID_dict:
        values = kc_to_tutorID_dict[key]
        for value in values:
            idx = value.find("__it_")
            if idx != -1:
                value = value[:idx]
            value = value.replace(":", "_")
            tutorID_to_kc_dict[value] = []
    
    for key in kc_to_tutorID_dict:
        values = kc_to_tutorID_dict[key]
        for value in values:
            idx = value.find("__it_")
            if idx != -1:
                value = value[:idx]
            value = value.replace(":", "_")
            tutorID_to_kc_dict[value].append(key)
        
    for key in tutorID_to_kc_dict:  
        tutorID_to_kc_dict[key] = pd.unique(tutorID_to_kc_dict[key]).tolist()

    return tutorID_to_kc_dict

def get_village_specific_bkt_params(kc_list_spaceless, uniq_student_ids, student_id_to_village_map, villages, path=''):
    
    num_skills = len(kc_list_spaceless)
    num_students = len(uniq_student_ids)

    init_know = np.ones((num_students, num_skills)) * 0.2
    init_learn = np.ones((num_students, num_skills)) * 0.6
    init_slip = np.ones((num_students, num_skills)) * 0.1
    init_guess = np.ones((num_students, num_skills)) * 0.3
    init_forget = np.zeros((num_students, num_skills))
    village_to_bkt_params = {}

    for village in villages:
        village_to_bkt_params[village] = np.zeros((num_skills, 4))
        params_file_name = path + "Data/village_" + village + "/params.txt"
        params_file = open(params_file_name, "r")
        contents = params_file.read().split('\n')[1:]

        for line in contents:
            line = line.split('\t')
            
            skill_name_spaceless = line[0]
            skill_idx = kc_list_spaceless.index(skill_name_spaceless)
            knew = float(line[1])
            learn = float(line[2])
            guess = float(line[3])
            slip = float(line[4])

            village_to_bkt_params[village][skill_idx][0] = knew
            village_to_bkt_params[village][skill_idx][1] = learn
            village_to_bkt_params[village][skill_idx][2] = slip
            village_to_bkt_params[village][skill_idx][3] = guess

    for student_id in uniq_student_ids:
        student_num = uniq_student_ids.index(student_id)
        village = student_id_to_village_map[student_id]
        village_bkt_params = village_to_bkt_params[village]
        
        for skill_num in range(num_skills):
            init_know[student_num][skill_num]   = village_bkt_params[skill_num][0]
            init_learn[student_num][skill_num]  = village_bkt_params[skill_num][1]
            init_slip[student_num][skill_num]   = village_bkt_params[skill_num][2]
            init_guess[student_num][skill_num]  = village_bkt_params[skill_num][3]
    
    return init_know, init_learn, init_slip, init_guess, init_forget

def get_student_specific_bkt_params(kc_list_spaceless, uniq_student_ids, student_id_to_village_map, path=''):
    
    num_skills = len(kc_list_spaceless)
    num_students = len(uniq_student_ids)
    
    # get params for the village here 
    init_know = np.ones((num_students, num_skills)) * 0.2
    init_learn = np.ones((num_students, num_skills)) * 0.4
    init_slip = np.ones((num_students, num_skills)) * 0.1
    init_guess = np.ones((num_students, num_skills)) * 0.3
    init_forget = np.zeros((num_students, num_skills))

    for student_id in uniq_student_ids:
        student_num = uniq_student_ids.index(student_id)
        
        if student_id == 'new_student':
            continue

        path_to_student_specific_params_file = path + "bkt_params/" + student_id + "_params.txt"
        # order: knew learn slip guess
        file = open(path_to_student_specific_params_file, "r")
        lines = file.read().split('\n')[1:]

        for i in range(len(lines)):
            if i == len(lines) - 1:
                # last line of file is empty line
                break
            line = lines[i].split('\t')
            skill_name_spaceless = line[0]
            skill_num = kc_list_spaceless.index(skill_name_spaceless)
            
            knew    = float(line[1])
            learn   = float(line[2])
            slip    = float(line[3])
            guess   = float(line[4])

            init_know[student_num][skill_num]    = knew
            init_learn[student_num][skill_num]   = learn
            init_slip[student_num][skill_num]    = slip
            init_guess[student_num][skill_num]   = guess

    return init_know, init_learn, init_slip, init_guess, init_forget

def get_bkt_params(kc_list_spaceless, uniq_student_ids, student_id_to_village_map, villages, subscript="student_specific", path=''):

    num_skills = len(kc_list_spaceless)
    num_students = len(uniq_student_ids)

    know = np.ones((num_students, num_skills)) * 0.1
    learn = np.ones((num_students, num_skills)) * 0.3
    slip = np.ones((num_students, num_skills)) * 0.2
    guess = np.ones((num_students, num_skills)) * 0.25
    forget = np.zeros((num_students, num_skills))

    if subscript == "student_specific":
        know, learn, slip, guess, forget = get_student_specific_bkt_params(kc_list_spaceless, uniq_student_ids, student_id_to_village_map, path)
    elif subscript == "village_specific":
        know, learn, slip, guess, forget = get_village_specific_bkt_params(kc_list_spaceless, uniq_student_ids, student_id_to_village_map, villages, path)
    else:
        print("Bad value for variable 'subscript' at helper.get_bkt_params()")
    
    print("Using", subscript, "BKT Subscripts")

    params_dict = {
        'know': know.tolist(),
        'learn': learn.tolist(),
        'slip': slip.tolist(),
        'guess': guess.tolist(),
        'forget': forget.tolist()
    }

    return params_dict

def get_col_vals_from_df(df, col_name, unique=False):
    """
        If unique is False gets all values of a column in the row-wise order and returns as a list.
        If unique is set to True, it returns unique values found under column "col_name" as a list
    """
    values = df[col_name].values.ravel()
    
    if unique == False:
        return values.tolist()
    
    elif unique == True:
        return pd.unique(values).tolist()
    
    else:
        print("ERROR in helper.get_col_vals_from_df()")
    
    return None

def get_kc_list_from_cta_table(cta_df):
    cta_columns = cta_df.columns.tolist()
    kc_list = cta_columns
    return kc_list

def get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df):
    """
        Gets unique tutor_ids from CTA table and also maps KC to tutorIDs that exercise a KC
        Removes NaN valued cells and cells that have "Column..."
    """
    cta_tutor_ids = []
    for kc in kc_list:
        if kc not in (cta_df.columns.tolist()):
            kc_to_tutorID_dict[kc] = []
        else:
            col_values = cta_df[[kc]].values.ravel()
            remove_idx = []
            for i in range(len(col_values)):
                if(isinstance(col_values[i], str) == False):
                    col_values[i] = str(col_values[i])
                if col_values[i].lower() == 'nan' or col_values[i].lower()[:6] == 'column':
                    remove_idx.append(i)
    
            col_values = np.delete(col_values, remove_idx)
        
            for i in range(len(col_values)):
                idx = col_values[i].find("__it_")
                if idx != -1:
                    col_values[i] = col_values[i][:idx]
            kc_to_tutorID_dict[kc] = col_values
            for val in col_values:
                cta_tutor_ids.append(val)
    
    cta_tutor_ids = pd.unique(np.array(cta_tutor_ids))
    return cta_tutor_ids, kc_to_tutorID_dict

def get_skill_groups_info(tutorID_to_kc_dict, kc_list):
    
    uniq_skill_groups = []
    skill_group_to_activity_map = {}

    for key in tutorID_to_kc_dict:
        skills = tutorID_to_kc_dict[key]
        skill_idxs = []
        for skill in skills:
            idx = kc_list.index(skill)
            skill_idxs.append(idx)
        if skill_idxs not in uniq_skill_groups:
            uniq_skill_groups.append(skill_idxs)
    
    for group in uniq_skill_groups:
        skill_group_to_activity_map[str(uniq_skill_groups.index(group))] = []

    for key in tutorID_to_kc_dict:
        skills = tutorID_to_kc_dict[key]
        skill_group = []
        for skill in skills:
            idx = kc_list.index(skill)
            skill_group.append(idx)
        skill_group_to_activity_map[str(uniq_skill_groups.index(skill_group))].append(key)
    
    # list, dict. The latter maps skill group '0' to activities that fall under this skill group
    return uniq_skill_groups, skill_group_to_activity_map

def get_underscore_to_colon_tutor_id_dict(cta_tutor_ids):
    
    underscore_to_colon_tutor_id_dict = {}

    for i in range(len(cta_tutor_ids)):
        underscored_id = cta_tutor_ids[i].replace(":", "_")
        colon_id = cta_tutor_ids[i]
        underscore_to_colon_tutor_id_dict[underscored_id] = colon_id
        cta_tutor_ids[i] = underscored_id
    
    return underscore_to_colon_tutor_id_dict

def get_uniq_activities(cta_tutor_ids, underscore_to_colon_tutor_id_dict):
    uniq_activities = []
    for tutor_id in cta_tutor_ids:
        if underscore_to_colon_tutor_id_dict[tutor_id] not in uniq_activities:
            uniq_activities.append(underscore_to_colon_tutor_id_dict[tutor_id])
    return uniq_activities
    




# Variable inits functions
def init_kc_to_tutorID_dict(kc_list):

    kc_to_tutorID_dict = {}
    for kc in kc_list:
        kc_to_tutorID_dict[kc] = []
    
    return kc_to_tutorID_dict

def init_item_learning_progress(n, u, uniq_item_student_ids):
    item_learning_progress = {}

    for i in range(n):
        student_id = uniq_item_student_ids[i]
        item_learning_progress[(student_id)] = np.array([np.zeros((u))]).T.tolist()
    
    return item_learning_progress

def init_act_student_id_to_number_map(n, u, activity_student_ids, act_student_id_to_number_map, knows):
    activity_learning_progress = {}
    for i in range(n):
        student_id = activity_student_ids[i]
        activity_learning_progress[(student_id)] = [knows[0].tolist()]
        act_student_id_to_number_map[student_id] = i 
    
    return activity_learning_progress, act_student_id_to_number_map