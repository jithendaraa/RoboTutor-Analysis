import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def rmse(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    
    squared_error = (a - b)**2
    mse = np.mean(squared_error)
    rmse_val = mse**0.5
    return rmse_val

# ITEM BKT is a binary BKT Model
class ItemBKT:
    # value of knew is always knew @ time (timestep)
    # g -> guess, s->slip, l->learn, f->forget, k->know: all of these are n*u arrays where we have n students and u skills or items
    # timestep-> n*u array which tells the current timestep of student n for skill u
    # KC subtest -> skills as per CTA table in 'Data/CTA.xlsx'
    """
        Functions
        ---------
        __init__()            : Initialises the class and sets class attributes
        update_per_obs()      : Does the step wise BKT update for "one observation"
        update()              : Does step wise BKT update for some number of observations by calling the update_per_obs() function
        predict_p_correct()   : After fitting, this returns P(Correct) of getting an item correct based on learnt BKT params 
    """
    def __init__(self, know, guess, slip, learn, KC_subtest, forget, num_students, num_skills, item_learning_progress):
        """
            Variables
            ---------
            self.know       --> an (n x u) matrix where self.know[i][j] is P(K @0) or prior know before any opportunities of student i, for skill j
            self.guess      --> an (n x u) matrix where self.guess[i][j] is P(Guess) of student i, for skill j. For an item with 4 choices this could be 0.25
            self.slip       --> an (n x u) matrix where self.slip[i][j] is P(Slip) of student i, for skill j. 
            self.learn      --> an (n x u) matrix where self.learn[i][j] is P(Learn) or P(Transit) of student i, for skill j. A fast learner will have a high P(learn)
            self.forget     --> an (n x u) matrix where self.foget[i][j] is P(Forget) of student i, for skill j. Usually always assumed to be 0.
            KC_subtest      --> list (of length num_skills) of all skills according to the CTA table 
            self.n          --> total students we are concerned about
            self.u          --> total skills involved
            self.timestep   --> an (n x u) matrix where self.timestep[i][j] tells the #opportunities student i has on skill j, initially this is np.zeros((n, u))
        """
        self.know = know
        self.guess = guess
        self.slip = slip
        self.learn = learn
        self.forget = forget
        self.KC_subtest = KC_subtest
        self.n = num_students
        self.u = num_skills
        self.timestep = np.zeros((self.n, self.u))
        self.item_learning_progress = item_learning_progress

    def update_per_obs(self, observation, i, j, item_learning_progress, transac_student_ids):
        """
            Description - update_per_obs()
            ----------------------
                Function to do the bayesian update per observation per skill. 
                Calc P(Know @t+1 | obs@ @t+1) based on P(Know @t), guess, slip, observation etc. 
                P(Know @t+1 | obs=CORRECT) = P(Know @t) * P(no slip)/( P(K @t)*P(no slip) + P(K @t)' * P(guess) )
                P(Know @t+1 | obs=INCORRECT) = P(Know @t) * P(slip)/( P(K @t)*P(slip) + P(K @t)' * P(guess)' )
                Then Calc P(Know @t+1) based on P(Learn) 
                P(Know @t+1) = P(Know @t+1 | obs) * P(no_forget) + P(Know @t+1 | obs)' * P(Learn)

            Parameters
            ----------
            observation: type boolean True or False. 
                         True -> item response by student was CORRECT
                         False -> item response by student was INCORRECT
        
            i:  type int 
                ith index of transac_student_ids will give the "Anon Student ID" associated with observation
        
            j:  type int
                jth index of self.KC_subtest will give the "KC (subtest)" associated with this observation
        
            learning_progress: dict with key as "Anon Student ID" 
                                value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                                shape of value: (u, *)

            transac_student_ids: list of unique "Anon Student Ids" from transaction table maintaining order of first appearance in the transactions table

            Returns
            -------
            learning_progress: type dict
                                After the BKT update, we get P(K @t+1) for skill j student transac_student_ids[i]
                                We append this P(K), the posterior, to learning_progress["Anon Student Ids"][j] before returning
            
        """
        t = self.timestep[i][j]
        anon_student_id = transac_student_ids[i]

        prior_know = self.know[i][j]
        prior_not_know = 1.0 - prior_know

        slip = self.slip[i][j]
        no_slip = 1.0 - slip

        guess = self.guess[i][j]
        no_guess = 1.0 - guess

        learn = self.learn[i][j]
        no_learn = 1.0 - learn

        forget = self.forget[i][j]
        no_forget = 1.0 - forget

        # posterior_know_given_obs -> P(K @t+1)
        posterior_know_given_obs = prior_know

        if observation == True:
            correct = (prior_know * no_slip) + (prior_not_know * guess)
            posterior_know_given_obs = prior_know * no_slip / correct
        
        elif observation == False:
            wrong = (prior_know * slip) + (prior_not_know * no_guess)
            posterior_know_given_obs = prior_know * slip / wrong
        
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn

        # Increment opportunity count for student i skill j
        self.timestep[i][j] = t+1
        self.know[i][j] = posterior_know
        
        # Update item_learning_progress with posterior P(K)
        item_learning_progress[anon_student_id][j].append(posterior_know)
        return item_learning_progress

    def update(self, item_observations, student_ids, skills, item_learning_progress, transac_student_ids):
        """
            Description - update()
            -------------------
                Function to do the bayesian update for k observations.
                Done by calling update_per_obs() for every observation 

            Parameters
            ----------
            item_observations:  type [boolean] of length k
                                True -> item response by student was CORRECT
                                False -> item response by student was INCORRECT
        
            student_ids:    type [int] of length k
                            ith entry of item_observations corresponds to Anon Student Id = transac_student_ids[student_ids[i]]

            skills:     type 2d list [[int]] of shape (k, self.u) where self.u is the total num_skills
                        all values are either 1 or 0
                        for ith item observation, we look at skills[i] which is a list of len self.u
                        if skills[i][j] == 1,it means that item_observations[i] is linked with skill self.KC_subtest[j]; else no

            item_learning_progress: type dict with key as "Anon Student ID" 
                                    value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                                    shape of value: (u, *)
            
            transac_student_ids: list of unique "Anon Student Ids" from transaction table maintaining order of first appearance in the transactions table
            
            Returns
            -------
            item_learning_progress:  type dict
                                    Update item_learning_progress which is done by self.update()
            
        """
        print("UPDATING BASED ON OBSERVATIONS....")
        k = len(item_observations)
        # For each attempt in item_observations
        for i in range(k):
            # For each skill
            for skill in skills[i]:
                # If this item obsevation exercises the skill; bayesian update is done for "every skill" that is exercised by this item; Perform a BKT update 
                if item_observations[i].upper() == "CORRECT":
                    self.update_per_obs(True, student_ids[i], skill, item_learning_progress, transac_student_ids)
                else:
                    self.update_per_obs(False, student_ids[i], skill, item_learning_progress, transac_student_ids)
        
        return self.item_learning_progress
                
    def predict_p_correct(self, student_id, skills):
        """
            Description - predict_p_correct()
            ---------------------------------
                Function to do predict P(correct) based on current BKT parameter values; usually called after fitting
            
            Parameters
            ----------
            student_id  --> type int    
                            index of student in transac_student_ids whose P(Correct) we need to predict

            skills       --> type [int]
                            index of skills into self.KC_subtest whose P(Correct) we need to predict
        """
        print("PREDICTING P(Correct)....")
        i = student_id
        correct = 1.0

        # Independent subskills performance prediction
        for skill in skills:
            j = skill
            p_know = self.know[i][j]
            p_not_know = 1.0 - p_know
            p_guess = self.guess[i][j]
            p_slip = self.slip[i][j]
            p_not_slip = 1.0 - p_slip

            # Independent subskill performance prediction
            correct = correct * ((p_know * p_not_slip) + (p_not_know * p_guess))
            
            # Weakest-subskill performance prediction 
            # correct = min(correct, (p_know * p_not_slip) + (p_not_know * p_guess))

        return correct

# ActivityBKT is a non-binary BKT Model
class ActivityBKT:
    # value of knew is always knew @ time (timestep)
    # g -> guess, s->slip, l->learn, f->forget, k->know: all of these are n*u arrays where we have n students and u skills or items
    # timestep-> n*u array which tells the current timestep of student n for skill u
    # KC subtest -> skills as per CTA table in 'Data/CTA.xlsx'
    """
        Functions
        ---------
        __init__()            : Initialises the class and sets class attributes
        update_per_obs()      : Does the activity wise BKT update for "one observation" of an activity 
        update()              : Does actiity wise BKT update for some number of observations by calling the update() function
        predict_p_correct()   : After fitting, this returns P(Correct) of getting an item correct based on learnt BKT params 
    """
    def __init__(self, know, guess, slip, learn, KC_subtest, students, forget, num_students, num_skills, activity_learning_progress):
        self.n = num_students
        self.u = num_skills
        self.timestep = np.zeros((self.n, self.u))
        self.know = know
        self.guess = guess
        self.slip = slip
        self.learn = learn
        self.forget = forget
        self.KC_subtest = KC_subtest
        self.students = students
        self.activity_learning_progress = activity_learning_progress
    
    """Function to do the bayesian update, ie calc P(Know @t+1 | obs @t+1) based on P(Know @t), guess slip etc. 
       Use this estimate along with P(Learn) to find P(Know @t+1)
    """
    def update_per_obs(self, observation, i, j):

        """
            Description - update_per_obs()
            ----------------------
                Function to do the activity wise bayesian update per observation per skill. 
                Calc P(Know @t+1 | obs@ @t+1) based on P(Know @t), guess, slip, observation etc. 
                P(Know @t+1 | obs as %correct) = (%correct * P(Know @t) * P(no slip)/( P(K @t)*P(no slip) + P(K @t)' * P(guess) )) + \
                                                (1 - %correct) * (P(Know @t) * P(slip)/( P(K @t)*P(slip) + P(K @t)' * P(guess)' ))
                Then Calc P(Know @t+1) based on P(Learn) 
                P(Know @t+1) = P(Know @t+1 | obs) * P(no_forget) + P(Know @t+1 | obs)' * P(Learn)

            Parameters
            ----------
            observation: type [float]. 
        
            i:  type int 
                ith index of activity_student_ids will give the "Unique_Child_ID_1" associated with observation
        
            j:  type int
                jth index of self.KC_subtest will give the "KC (subtest)" associated with this observation
        
            learning_progress: dict with key as "Unique_Child_ID_1" 
                                value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                                shape of value: (u, *)

            act_student_ids: list of unique "Unique_Child_ID_1" from activity table maintaining order of first appearance in the activity table

            Returns
            -------
            learning_progress: type dict
                                After the BKT update, we get P(K @t+1) for skill j student act_student_ids[i]
                                We append this P(K), the posterior, to learning_progress["Unique_Child_ID_1"][j] before returning
        """

        t = self.timestep[i][j]

        percent_correct = observation
        percent_incorrect = 1.0 - percent_correct

        prior_know = self.know[i][j]
        prior_not_know = 1.0 - prior_know

        slip = self.slip[i][j]
        no_slip = 1.0 - slip

        guess = self.guess[i][j]
        no_guess = 1.0 - guess

        learn = self.learn[i][j]
        no_learn = 1.0 - learn

        forget = self.forget[i][j]
        no_forget = 1.0 - forget

        # posterior_know_given_obs -> P(K @t+1)
        posterior_know_given_obs = prior_know

        correct = prior_know * no_slip + prior_not_know * guess
        wrong = prior_know * slip + prior_not_know * no_guess
        
        posterior_know_given_obs = (percent_correct * ((prior_know * no_slip) / correct )) + \
                                    (percent_incorrect * ((prior_know * slip) / wrong))
        
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn

        self.timestep[i][j] = t+1
        self.know[i][j] = posterior_know
        

    def update(self, activity_observations, student_ids, skills):
        """
            Description - update()
            -------------------
                Function to do the bayesian update for k observations.
                Done by calling update() for every observation 

            Parameters
            ----------
            activity_observations:  type [float] of length k
        
            student_ids:    type [int] of length k
                            ith entry of item_observations corresponds to Unique_Child_ID_1 = activity_student_ids[student_ids[i]]

            skills:     type 2d list [[int]] of shape (k, self.u) where self.u is the total num_skills
                        skills[i] contains index of skills exercised by ith observations. Indexes into self.KC_subtest

            activity_learning_progress: type dict with key as "Unique_Child_ID_1" 
                                        value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                                        shape of value: (u, *)
            
            activity_student_ids: list of unique "Unique_Child_ID_1" from activity table maintaining order of first appearance in the activity table
            
            Returns
            -------
            activity_learning_progress:  type dict
                                        Update activity_learning_progress which is done by self.update()
        """

        # print("UPDATING P(Know) BASED ON OBSERVATIONS....")
        k = len(activity_observations)
        # For each attempt in item_observations
        for i in range(k):
            # For each skill
            # knows = self.activity_learning_progress[self.students[student_ids[i]]].copy()
            # latest_know = knows[-1]
            for j in range(len(skills[i])):
            # If this activity obsevation exercises the skill; bayesian update is done for "every skill" that is exercised by this activity, perform a BKT update
                skill_num = skills[i][j]
                student_num = student_ids[i]
                self.update_per_obs(activity_observations[i], student_num, skill_num)
                # latest_know[skill] = posterior
                if j == len(skills[i]) - 1: 
                    self.activity_learning_progress[self.students[student_num]].append(self.know[student_num].tolist())
        
    def predict_percent_correct(self, student_ids, skills, actual_observations=None):
        # print("PREDICTING P(Correct)....")

        correct_preds = []
        min_correct_preds = []
        n = len(student_ids)

        for k in range(n):
            i = student_ids[k]
            correct = 1.0
            min_correct = 1.0
            for skill in skills[k]:
                j = skill
                p_know = self.know[i][j]
                p_not_know = 1.0 - p_know
                p_guess = self.guess[i][j]
                p_slip = self.slip[i][j]
                p_not_slip = 1.0 - p_slip
                # Independent subskills prediction
                correct = correct * ((p_know * p_not_slip) + (p_not_know * p_guess))
                # Weakest-subskill performance prediction 
                min_correct = min(min_correct, (p_know * p_not_slip) + (p_not_know * p_guess))
            
            # print("PREDICTED (Full responsibility): ", correct, skills[k])
            # print("PREDICTED (Blame weakest): ", min_correct, skills[k])
            if actual_observations != None:
                for skill in skills[k]:
                    self.update_per_obs(actual_observations[k], student_ids[k], skill)
            correct_preds.append(correct)
            min_correct_preds.append(min_correct)
            
        if actual_observations != None:
            actual_observations = np.array(actual_observations)
            correct_preds = np.array(correct_preds)
            min_correct_preds = np.array(min_correct_preds)
            correct_preds_rmse = rmse(actual_observations, correct_preds)
            min_correct_preds_rmse = rmse(actual_observations, min_correct_preds)
            print("FULL RESPONSIBILITY P(Correct) prediction RMSE: ", correct_preds_rmse)
            print("BLAME WORST P(Correct) prediction RMSE: ", min_correct_preds_rmse)
        
            return correct_preds_rmse, min_correct_preds_rmse
        
        return correct_preds, min_correct_preds
                   
def plot_learning(learning_progress, student_ids, timesteps, new_student_avg, student_avgs, algo):
    """
        Description - plot_learning()
        -----------------------------

        Parameters
        ----------
        learning_progress:  type dict with key as "Anon Student ID" or "Unique_Child_ID_1"; could be activity_learning_progress or item_learning_progress
                            value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                            shape of value: (u, *)

        student_ids     :   List of indices of students; transac_student_ids[student_id] gives "Anon Student ID" and acitivity_student_ids[student_id] gives "Unique_Child_ID_1"
        timesteps       :   int. Total #opportunities we want to plot for
        new_student_avg :   List. Avg P(Know) of "new_student" after every opportunity. This is usually the student that has learnt from the RL Policy
        student_avgs    :   2DList. student[avgs[i]] contains avg P(Know) of student with "UNIQUE_CHILD_ID_1" = student_ids[i] after every opportunity. These are
                            avg P(Know)'s after learning from RoboTutor's learning policy. Inferred from observations in the activity table using train_on_obs() in student_simulator.py
        Plots
        -----
        Avg P(Know) across skills after every opportunity        
        Returns
        -------
        No return
    """
    colors = ["red", "green", "yellow", "purple", "blue", "violet", "orange", "brown"]
    for i in range(len(student_ids)):
        student_id = student_ids[i]
        x = []
        y = []
        for j in range(len(learning_progress[student_id])):
            p_know = learning_progress[student_id][j]
            p_avg_know = np.mean(np.array(p_know))
            y.append(p_avg_know)
            x.append(j+1)
        plt.plot(x[:timesteps], y[:timesteps], label=student_id, color=colors[i])
    
    x = np.arange(1, len(new_student_avg) + 1).tolist()
    plt.plot(x, new_student_avg, label="RL Agent", color="black")
    plt.legend()
    plt.xlabel("# Opportunities")
    plt.ylabel("Avg P(Know) across 18 skills")
    plt.title("Avg P(Know) vs. #opportunities")
    plt.savefig("plots/" + algo + '_results.jpg')
    # plt.show()

def read_transac_table(path_to_transac_table, full_df=False):
    # Reads transac table and retuns it as a df after dropping some columns that areent required for our use
    transac_df = pd.read_csv(path_to_transac_table, sep='\t')
    # Make Anon Student Id uniform data type
    transac_df = transac_df.astype({"Anon Student Id": str})
    
    if full_df == False:
        drop_cols = ["Row", "Sample Name", "Session Id","Time","Problem Start Time","Time Zone", "Duration (sec)", "Student Response Type", "Student Response Subtype", "Tutor Response Type", "Tutor Response Subtype", "Selection", "Action", "Feedback Text", "Feedback Classification", "Help Level", "Total Num Hints", "School", "Class", "CF (File)", "CF (Hiatus sec)", "CF (Original DS Export File)","CF (Unix Epoch)","CF (Village)","Event Type","CF (Student Used Scaffold)","CF (Robotutor Mode)","KC Category (Single-KC)","KC Category (Unique-step)","CF (Child Id)","CF (Date)","CF (Placement Test Flag)","CF (Week)","Transaction Id","Problem View","KC (Single-KC)","KC (Unique-step)", "CF (Activity Finished)", "CF (Activity Started)", "CF (Attempt Number)","CF (Duration sec)", "CF (Expected Answer)", "CF (Matrix)", "CF (Matrix Level)", "CF (Matrix Order)", "CF (Original Order)","CF (Outcome Numeric)", "CF (Placement Test User)", "CF (Problem Number)", "CF (Session Sequence)", "CF (Student Chose Repeat)", "CF (Total Activity Problems)", "CF (Tutor Sequence Session)", "CF (Tutor Sequence User)","Input", "Is Last Attempt", "Attempt At Step", "Step Name"]
        transac_df = transac_df.drop(columns=drop_cols)

    return transac_df

def read_cta_table(path_to_cta_table):
    cta_df = pd.read_excel(path_to_cta_table)
    return cta_df

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

# given a string remove "__it_" and return string.
def remove_iter_suffix(tutor):
    if isinstance(tutor, list):
        
        for i in range(len(tutor)):
            idx = tutor[i].find("__it_")
            if idx != -1:
                tutor[i] = tutor[i][:idx]
        return tutor
    
    elif isinstance(tutor, str):
        idx = tutor.find("__it_")
        if idx != -1:
            tutor = tutor[:idx]
        return tutor
    
    else:
        print("ERROR: remove_iter_suffix")
    
    return None

def get_kc_list_from_cta_table(cta_df):
    cta_columns = cta_df.columns.tolist()
    kc_list = cta_columns
    return kc_list

def init_kc_to_tutorID_dict(kc_list):
    kc_to_tutorID_dict = {}
    for kc in kc_list:
        kc_to_tutorID_dict[kc] = []
    
    return kc_to_tutorID_dict

def get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df):
    """
        Gets unique tutor_ids from CTA table and also maps KC to tutorIDs that exercise a KC
        Removes NaN valued cells and cells that have "Column..."
    """
    cta_tutor_ids = []
    for kc in kc_list:
        col_values = cta_df[[kc]].values.ravel()
        remove_idx = []
        for i in range(len(col_values)):
            if(isinstance(col_values[i], str) == False):
                col_values[i] = str(col_values[i])
            if col_values[i].lower() == 'nan' or col_values[i].lower()[:6] == 'column':
                remove_idx.append(i)
    
        col_values = np.delete(col_values, remove_idx)
        kc_to_tutorID_dict[kc] = col_values
        for val in col_values:
            cta_tutor_ids.append(val)
    
    cta_tutor_ids = pd.unique(np.array(cta_tutor_ids))
    return cta_tutor_ids, kc_to_tutorID_dict

def init_underscore_to_colon_tutor_id_dict(cta_tutor_ids):
    
    underscore_to_colon_tutor_id_dict = {}

    for i in range(len(cta_tutor_ids)):
        underscored_id = cta_tutor_ids[i].replace(":", "_")
        colon_id = cta_tutor_ids[i]
        underscore_to_colon_tutor_id_dict[underscored_id] = colon_id
        cta_tutor_ids[i] = underscored_id
    
    return underscore_to_colon_tutor_id_dict

def init_tutorID_to_kc_dict(kc_to_tutorID_dict):
    """
        Init and populate tutorID_to_kc_dict
    """
    tutorID_to_kc_dict = {}
    for key in kc_to_tutorID_dict:
        values = kc_to_tutorID_dict[key]
        for value in values:
            tutorID_to_kc_dict[value] = []
    
    for key in kc_to_tutorID_dict:
        values = kc_to_tutorID_dict[key]
        for value in values:
            tutorID_to_kc_dict[value].append(key)
        
    for key in tutorID_to_kc_dict:
        tutorID_to_kc_dict[key] = pd.unique(tutorID_to_kc_dict[key]).tolist()

    return tutorID_to_kc_dict

def init_item_learning_progress(n, u, uniq_item_student_ids):
    item_learning_progress = {}

    for i in range(n):
        student_id = uniq_item_student_ids[i]
        item_learning_progress[(student_id)] = np.array([np.zeros((u))]).T.tolist()
    
    return item_learning_progress

# def init_student_id_to_number_map(df, type):
    
#     student_id_to_number_map = {}
#     if type=="transac":
#         uniq_student_ids = pd.unique(df["Anon Student Id"].values.ravel()).tolist()
#     elif type=="activity":
#         uniq_student_ids = pd.unique(df["Unique_Child_ID_1"].values.ravel()).tolist()
    
#     uniq_student_ids.append("new_student")

#     for i in range(len(uniq_student_ids)):
#         student_id = uniq_student_ids[i]
#         student_id_to_number_map[student_id] = i
    
#     return student_id_to_number_map

def init_skill_to_number_map(kc_list):
    skill_to_number_map = {}
    for i in range(len(kc_list)):
        skill = kc_list[i]
        skill_to_number_map[skill] = i
    return skill_to_number_map

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

    return item_observations, item_student_ids, item_skills, uniq_transac_student_ids

def init_act_student_id_to_number_map(n, u, activity_student_ids, act_student_id_to_number_map, knows):
    activity_learning_progress = {}
    for i in range(n):
        student_id = activity_student_ids[i]
        activity_learning_progress[(student_id)] = [knows[0].tolist()]
        act_student_id_to_number_map[student_id] = i 
    
    return activity_learning_progress, act_student_id_to_number_map


def extract_activity_table(activity_df, act_student_id_to_number_map, tutorID_to_kc_dict, skill_to_number_map, underscore_to_colon_tutor_id_dict):
    """
        Reads actiivty table, gets necessary data from it and gets all details that are necessary to do the activityBKT update.
        Returns observations, student_ids, skills involved with each of these attempts, num_corrects and num_attempts
        ith row of each of these lists give info about the ith opportunity or corresponds to ith row of activity_df
    """
    activity_observations = []
    student_ids = []

    student_ids = get_col_vals_from_df(activity_df, "Unique_Child_ID_1", unique=False)
    activity_names = get_col_vals_from_df(activity_df, "ActivityName", unique=False)
    num_rows = activity_df.shape[0]
    activity_skills = [0] * num_rows

    num_corrects = get_col_vals_from_df(activity_df, "total_correct_attempts", unique=False)
    num_attempts = get_col_vals_from_df(activity_df, "#attempts", unique=False)

    for j in range(num_rows):
        percent_correct = num_corrects[j]/num_attempts[j]
        # print(num_attempts[j], "------", num_corrects[j])
        activity_observations.append(percent_correct)
        student_ids[j] = act_student_id_to_number_map[student_ids[j]]

        activity_name = remove_iter_suffix(activity_names[j])

        if activity_name in tutorID_to_kc_dict:
            activity_skills[j] = tutorID_to_kc_dict[activity_name]
    
        elif activity_name == "activity_selector":
            underscored_id = activity_df["TutorID"].values.ravel().tolist()[j].strip()
            underscored_id = remove_iter_suffix(underscored_id)
            colon_id = underscore_to_colon_tutor_id_dict[underscored_id]
            activity_skills[j] = tutorID_to_kc_dict[colon_id]
        
        else:
            print("ACTIVITY NAME: ", activity_name, "is NOT VALID")
            return None

        res = []
        for skill in activity_skills[j]:    
            res.append(skill_to_number_map[skill])
        activity_skills[j] = res

    return activity_observations, student_ids, activity_skills, num_corrects, num_attempts

# def sample_activity(mu, sigma, matrix, row, lit_nan_idxs):
#     new_col = math.floor(np.random.normal(mu, sigma, 1))
#     while lit_nan_idxs[row] <= new_col or (isinstance(matrix[row][new_col], str) == False and math.isnan(matrix[row][new_col])):
#         print(new_col, row)
#         new_col -= lit_nan_idxs[row]
#         row += 1
    
#     col = new_col
#     activity = matrix[row][col]
#     return row, col, activity

def get_activity_matrix(PATH_TO_ACTIVITY_DIFFICULTY, LITERACY_SHEET_NAME, MATH_SHEET_NAME, STORIES_SHEET_NAME):
    """
        Takes paths and sheet names as params and returns the 3 activity matrices
    """
    xls = pd.ExcelFile(PATH_TO_ACTIVITY_DIFFICULTY)

    # Difficulty with levels as rows
    literacy_matrix = pd.read_excel(xls, LITERACY_SHEET_NAME)[1:].values.tolist()
    math_matrix = pd.read_excel(xls, MATH_SHEET_NAME).values.tolist()
    stories_matrix = pd.read_excel(xls, STORIES_SHEET_NAME).values.tolist()

    return literacy_matrix, math_matrix, stories_matrix

def get_proba(action, activityName, tutorID_to_kc_dict, skill_to_number_map, p_know):
    # returns list of P(Know) for only those skills that are exercised by activityName
    proba = []
    
    skillNames = tutorID_to_kc_dict[activityName]
    skillNums = []
    for skillName in skillNames:
        skillNums.append(skill_to_number_map[skillName])
    
    for skill in skillNums:
        proba.append(p_know[skill])
    
    return proba

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

def read_data():
    """
        Reads and returns some useful data from CTA Table and activity_table.
    """
    cta_df = read_cta_table("Data/CTA.xlsx")
    kc_list = get_kc_list_from_cta_table(cta_df)
    u = len(kc_list)
    num_skills = u
    kc_to_tutorID_dict = init_kc_to_tutorID_dict(kc_list)
    cta_tutor_ids, kc_to_tutorID_dict = get_cta_tutor_ids(kc_to_tutorID_dict, kc_list, cta_df)
    tutorID_to_kc_dict = init_tutorID_to_kc_dict(kc_to_tutorID_dict)
    uniq_skill_groups, skill_group_to_activity_map = get_skill_groups_info(tutorID_to_kc_dict, kc_list)
    # print("DATA READING DONE.....")
    return kc_list, num_skills, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map


def clear_files(algo, clear):
    # empties all txt files under algo + "_logs" folder if clear is set to True
    if clear == False:
        return
    log_folder_name = algo + "_logs"
    files = os.listdir(log_folder_name)
    text_files = []
    
    for file in files:
        if file[-3:] == "txt":
            text_files.append(file)
    
    for text_file in text_files:
        file = open(log_folder_name + "/" + text_file, "r+")
        file.truncate(0)
        file.close()