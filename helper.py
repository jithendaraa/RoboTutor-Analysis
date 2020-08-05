import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tqdm import tnrange, tqdm_notebook, tqdm

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
    def __init__(self, know, guess, slip, learn, kc_list, forget, num_students, num_skills, uniq_student_ids):
        """
            Variables
            ---------
            self.know       --> an (n x u) matrix where self.know[i][j] is P(K @0) or prior know before any opportunities of student i, for skill j
            self.guess      --> an (n x u) matrix where self.guess[i][j] is P(Guess) of student i, for skill j. For an item with 4 choices this could be 0.25
            self.slip       --> an (n x u) matrix where self.slip[i][j] is P(Slip) of student i, for skill j. 
            self.learn      --> an (n x u) matrix where self.learn[i][j] is P(Learn) or P(Transit) of student i, for skill j. A fast learner will have a high P(learn)
            self.forget     --> an (n x u) matrix where self.foget[i][j] is P(Forget) of student i, for skill j. Usually always assumed to be 0.
            kc_list         --> list (of length num_skills) of all skills according to the CTA table 
            self.n          --> total students we are concerned about
            self.u          --> total skills involved
            self.timestep   --> an (n x u) matrix where self.timestep[i][j] tells the #opportunities student i has on skill j, initially this is np.zeros((n, u))
        """
        self.know = know
        self.guess = guess
        self.slip = slip
        self.learn = learn
        self.forget = forget
        self.kc_list = kc_list
        self.n = num_students
        self.u = num_skills
        self.students = uniq_student_ids
        self.learning_progress = {}

        for i in range(self.n):
            student_id = self.students[i]
            self.learning_progress[student_id] = [self.know[i].copy()]

    def update_per_obs(self, observation, i, j):
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
                ith index of self.students will give the "Anon Student ID" associated with observation
        
            j:  type int
                jth index of self.kc_list will give the "KC (subtest)" associated with this observation

            Returns
            -------
            learning_progress: type dict
                                After the BKT update, we get P(K @t+1) for skill j student transac_student_ids[i]
                                We append this P(K), the posterior, to learning_progress["Anon Student Ids"][j] before returning
            
        """
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
        posterior_know_given_obs = None

        correct = (prior_know * no_slip) + (prior_not_know * guess)
        wrong = (prior_know * slip) + (prior_not_know * no_guess)

        if observation == 1:
            posterior_know_given_obs = (prior_know * no_slip / correct)
        elif observation == 0:
            posterior_know_given_obs = (prior_know * slip / wrong)
        
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn

        # Increment opportunity count for student i skill j
        self.know[i][j] = posterior_know

        
    def update(self, student_nums, skill_nums, corrects):

        num_rows = len(student_nums)

        for i in range(num_rows):
            student_num = student_nums[i]
            student_id = self.students[student_num]

            for skill_num in skill_nums[i]:
                self.update_per_obs(corrects[i], student_num, skill_num)
            
            self.learning_progress[student_id].append(self.know[student_num].copy())
            # print(self.know[student_num], np.mean(np.array(self.know[student_num])), skill_nums[i])


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
    def __init__(self, know, guess, slip, learn, KC_subtest, students, forget, num_students, num_skills, activity_learning_progress, know_act=None, guess_act=None, slip_act=None, learn_act=None, forget_act=None, difficulty=None):
        self.n = num_students
        self.u = num_skills
        self.num_acts = 1710
        self.timestep = np.zeros((self.n, self.u))
        self.timestep_act = np.zeros((self.n, self.num_acts))
        self.know = know
        self.guess = guess
        self.slip = slip
        self.learn = learn
        self.forget = forget
        self.KC_subtest = KC_subtest
        self.students = students
        self.learning_progress = activity_learning_progress
        self.know_act = know_act
        self.guess_act = guess_act
        self.slip_act = slip_act
        self.learn_act = learn_act
        self.forget_act = forget_act

        self.difficulty = difficulty
        self.coefficients = np.zeros((self.num_acts, self.u))
        
    def set_coefficients(self, uniq_activities):
        p_know = [0] * self.u
        
        for i in range(self.u):
            length_of_list = len(self.difficulty[i])
            normalising_constant = (length_of_list * (1+length_of_list))/2
            for act in uniq_activities:
                position = -1
                if act in self.difficulty[i]:
                    position = self.difficulty[i].index(act)
                
                coefficient = 1/length_of_list
                # coefficient = 1/length_of_list
                if position == -1:
                    coefficient = 0

                act_idx = uniq_activities.index(act)
                self.coefficients[act_idx][i] = coefficient
        
    def set_learning_progress(self, student_id, learning_progress, know):
        self.learning_progress[student_id] = learning_progress
        self.know[self.students.index(student_id)] = know
    
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
    
    def update_per_activity(self, observation, i, j):

        percent_correct = observation
        percent_incorrect = 1.0 - percent_correct
        prior_know = self.know_act[i][j]
        prior_not_know = 1.0 - prior_know
        slip = self.slip_act[i][j]
        no_slip = 1.0 - slip
        guess = self.guess_act[i][j]
        no_guess = 1.0 - guess
        learn = self.learn_act[i][j]
        no_learn = 1.0 - learn
        forget = self.forget_act[i][j]
        no_forget = 1.0 - forget

        # posterior_know_given_obs -> P(K @t+1)
        posterior_know_given_obs = prior_know
        correct = prior_know * no_slip + prior_not_know * guess
        wrong = prior_know * slip + prior_not_know * no_guess
        posterior_know_given_obs = (percent_correct * ((prior_know * no_slip) / correct )) + \
                                    (percent_incorrect * ((prior_know * slip) / wrong))
        posterior_know = (posterior_know_given_obs * no_forget) + (1 - posterior_know_given_obs) * learn
        self.know_act[i][j] = posterior_know
        
    def update_know_skills(self, activity_name, activity_num, student_num, uniq_activities):
        
        p_know = [0] * self.u
        
        for i in range(self.u):
            for act in uniq_activities:
                act_idx = uniq_activities.index(act)
                p_know[i] += (self.coefficients[act_idx][i] * self.know_act[student_num][act_idx])
        
        self.know[student_num] = p_know
        
    def update_by_activity(self, activity_observations, student_nums, activities, tutor_ids, underscore_to_colon_tutor_id_dict):
        k = len(activities)
        uniq_activities = []
        for tutor_id in tutor_ids:
            if underscore_to_colon_tutor_id_dict[tutor_id] not in uniq_activities:
                uniq_activities.append(underscore_to_colon_tutor_id_dict[tutor_id])
        
        for i in range(k):
            student_num = student_nums[i]
            student_id = self.students[student_num]
            print(i)
            activity_num = uniq_activities.index(activities[i])
            activity_name = activities[i]
            self.update_per_activity(activity_observations[i], student_num, activity_num)
            self.update_know_skills(activity_name, activity_num, student_num, uniq_activities)                
            self.learning_progress[student_id].append(self.know[student_num].tolist())

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
                   
def plot_learning(learning_progress, student_ids, timesteps, new_student_avg, algo):

    colors = ["red", "green", "yellow", "purple", "blue", "violet", "orange", "brown"]
    for i in range(len(student_ids)):
        student_id = student_ids[i]
        x = []
        y = []
        for j in range(len(learning_progress[student_id])):
            p_know = learning_progress[student_id][j]
            p_avg_know = np.mean(np.array(p_know))
            x.append(j+1)
            y.append(p_avg_know)
            # print(p_know, p_avg_know)
            if j>70:
                break
        plt.plot(x[:70], y[:70], label=student_id)
    
    # x = np.arange(1, len(new_student_avg) + 1).tolist()
    # plt.plot(x, new_student_avg, label="RL Agent", color="black")
    plt.legend()
    plt.xlabel("# Opportunities")
    plt.ylabel("Avg P(Know) across skills")
    plt.title("Avg P(Know) vs. #opportunities")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    # plt.savefig("plots/" + algo + '_results.jpg')
    plt.show()

def get_difficulties(num_skills, tutorID_to_kc_dict, kc_list):
    difficulty = np.zeros((num_skills, 1)).tolist()

    PATH_TO_ACTIVITY_DIFFICULTY = "Data/Code Drop 2 Matrices.xlsx"
    LITERACY_SHEET_NAME = 'Literacy (with levels as rows)'
    MATH_SHEET_NAME = 'Math (with levels as rows)'
    STORIES_SHEET_NAME = 'Stories'

    literacy_matrix, math_matrix, stories_matrix = get_activity_matrix(PATH_TO_ACTIVITY_DIFFICULTY, LITERACY_SHEET_NAME, MATH_SHEET_NAME, STORIES_SHEET_NAME)

    # remove nans in all 3 activity matrices
    for i in range(len(math_matrix)):
        row_vals = math_matrix[i]
        while isinstance(row_vals[len(row_vals) - 1], str) == False and math.isnan(row_vals[len(row_vals) - 1]):
            row_vals.pop(len(row_vals) - 1)
        math_matrix[i] = row_vals
    
    for i in range(len(stories_matrix)):
        row_vals = stories_matrix[i]
        while isinstance(row_vals[len(row_vals) - 1], str) == False and math.isnan(row_vals[len(row_vals) - 1]):
            row_vals.pop(len(row_vals) - 1)
        stories_matrix[i] = row_vals
    
    for i in range(len(literacy_matrix)):
        row_vals = literacy_matrix[i]
        while isinstance(row_vals[len(row_vals) - 1], str) == False and math.isnan(row_vals[len(row_vals) - 1]):
            row_vals.pop(len(row_vals) - 1)
        literacy_matrix[i] = row_vals

    math_list = [j for sub in math_matrix for j in sub] 
    stories_list = [j for sub in stories_matrix for j in sub] 
    literacy_list = [j for sub in literacy_matrix for j in sub] 

    for activity in math_list:
        skills = []
        for val in tutorID_to_kc_dict[activity]:
            skills.append(kc_list.index(val))
        for skill in skills:
            if activity not in difficulty[skill]:
                difficulty[skill].append(activity)
    
    for activity in stories_list:
        skills = []
        if activity == "story.hear::Garden_Song.1":
            activity = "story.hear::Garden_Song"
        elif activity == "story.hear::Safari_Song.1":
            activity = "story.hear::Safari_Song"

        for val in tutorID_to_kc_dict[activity]:
            skills.append(kc_list.index(val))
        for skill in skills:
            if activity not in difficulty[skill]:
                difficulty[skill].append(activity)
    
    for activity in literacy_list:
        skills = []
        for val in tutorID_to_kc_dict[activity]:
            skills.append(kc_list.index(val))
        for skill in skills:
            if activity not in difficulty[skill]:
                difficulty[skill].append(activity)

    for i in range(len(difficulty)):
        if len(difficulty[i]) > 1:
            difficulty[i] = difficulty[i][1:]

    return difficulty

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
    cta_df = pd.read_excel(path_to_cta_table).astype({'Quantifying': str})
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

def init_item_learning_progress(n, u, uniq_item_student_ids):
    item_learning_progress = {}

    for i in range(n):
        student_id = uniq_item_student_ids[i]
        item_learning_progress[(student_id)] = np.array([np.zeros((u))]).T.tolist()
    
    return item_learning_progress

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

def extract_activity_table(activity_df, act_student_id_to_number_map, kc_list):
    """
        Reads actiivty table, gets necessary data from it and gets all details that are necessary to do the activityBKT update.
        Returns observations, student_ids, skills involved with each of these attempts, num_corrects and num_attempts
        ith row of each of these lists give info about the ith opportunity or corresponds to ith row of activity_df
    """

    activity_skills = []

    activity_observations = get_col_vals_from_df(activity_df, "%correct", unique=False)
    student_ids = get_col_vals_from_df(activity_df, "Unique_Child_ID_1", unique=False)
    num_corrects = get_col_vals_from_df(activity_df, "total_correct_attempts", unique=False)
    num_attempts = get_col_vals_from_df(activity_df, "#net_attempts", unique=False)
    activity_names = get_col_vals_from_df(activity_df, "ActivityName", unique=False)
    kc_subtests = get_col_vals_from_df(activity_df, "KC (Subtest)", unique=False)
    
    student_nums = []
    for student_id in student_ids:
        student_nums.append(act_student_id_to_number_map[student_id])
    
    for kc_subtest in kc_subtests:
        activity_skills.append(kc_list.index(kc_subtest)) 

    return activity_observations, student_nums, activity_skills, num_corrects, num_attempts, activity_names

def get_activity_matrix(PATH_TO_ACTIVITY_DIFFICULTY, LITERACY_SHEET_NAME, MATH_SHEET_NAME, STORIES_SHEET_NAME):
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
    print("DATA READING DONE.....")
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

def get_spaceless_kc_list(kc_list):
    res = []

    for kc in kc_list:
        res.append(kc.replace(" ", ""))
    
    return res

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

def extract_step_transac(path_to_data, uniq_student_ids, kc_list_spaceless, student_id=None, train_split=1.0):

    student_ids = []
    student_nums = []
    skill_names = []
    skill_nums = []
    corrects = []

    df = pd.read_csv(path_to_data, delimiter='\t', header=None).astype({1: str})

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
    
    return student_ids, student_nums, skill_names, skill_nums, corrects, test_idx

def village_specific_params(num_skills, kc_list_spaceless, uniq_student_ids, num_students, student_id_to_village_map, villages):
    
    init_know = np.ones((num_students, num_skills)) * 0.2
    init_learn = np.ones((num_students, num_skills)) * 0.6
    init_slip = np.ones((num_students, num_skills)) * 0.1
    init_guess = np.ones((num_students, num_skills)) * 0.3
    init_forget = np.zeros((num_students, num_skills))
    village_to_bkt_params = {}

    for village in villages:
        village_to_bkt_params[village] = np.zeros((num_skills, 4))
        params_file_name = "Data/village_" + village + "/params.txt"
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

def student_specific_params(num_skills, kc_list_spaceless, uniq_student_ids, num_students, student_id_to_village_map, villages):
    init_know = np.ones((num_students, num_skills)) * 0.2
    init_learn = np.ones((num_students, num_skills)) * 0.6
    init_slip = np.ones((num_students, num_skills)) * 0.1
    init_guess = np.ones((num_students, num_skills)) * 0.3
    init_forget = np.zeros((num_students, num_skills))

    for student_id in uniq_student_ids:
        student_num = uniq_student_ids.index(student_id)
        path_to_student_specific_params_file = "bkt_params/" + student_id + "_params.txt"
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

def get_bkt_params(num_skills, kc_list_spaceless, uniq_student_ids, num_students, student_id_to_village_map, villages, subscript="student_specific"):

    know = np.ones((num_students, num_skills)) * 0.1
    learn = np.ones((num_students, num_skills)) * 0.3
    slip = np.ones((num_students, num_skills)) * 0.2
    guess = np.ones((num_students, num_skills)) * 0.25
    forget = np.zeros((num_students, num_skills))

    if subscript == "student_specific":
        know, learn, slip, guess, forget = student_specific_params(num_skills, kc_list_spaceless, uniq_student_ids, num_students, student_id_to_village_map, villages)
    elif subscript == "village_specific":
        know, learn, slip, guess, forget = village_specific_params(num_skills, kc_list_spaceless, uniq_student_ids, num_students, student_id_to_village_map, villages)
    else:
        print("IMPROPER SUBSCRIPT at helper.get_bkt_params()")
    
    print("Using", subscript, "BKT Subscripts")

    return know.tolist(), learn.tolist(), slip.tolist(), guess.tolist(), forget.tolist()

def sigmoid(x):
    return 1/(1+np.exp(-x))



