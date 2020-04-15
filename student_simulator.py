# Imports
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

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
        update()              : Does the step wise BKT update for "one observation"
        fit()                 : Does step wise BKT update for some number of observations by calling the update() function
        predict_p_correct()   : After fitting, this returns P(Correct) of getting an item correct based on learnt BKT params 
    """
    def __init__(self, know, guess, slip, learn, KC_subtest, forget, num_students, num_skills):
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

    def update(self, observation, i, j, learning_progress, transac_student_ids):
        """
            Description - update()
            ----------------------
                Function to do the bayesian update per observation. 
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

    def fit(self, item_observations, student_ids, skills, item_learning_progress, transac_student_ids):
        """
            Description - fit()
            -------------------
                Function to do the bayesian update for k observations.
                Done by calling update() for every observation 

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
        print("FITTING BASED ON OBSERVATIONS....")
        k = len(item_observations)
        # For each attempt in item_observations
        for i in range(k):
            # For each skill
            for j in range(self.u):
                # If this item exercises the skill
                if(skills[i][j] == 1):
                    # Perform a BKT update
                    if item_observations[i].upper() == "CORRECT":
                        item_learning_progress = self.update(True, student_ids[i], j, item_learning_progress, transac_student_ids)
                    else:
                        item_learning_progress = self.update(False, student_ids[i], j, item_learning_progress, transac_student_ids)
        
        return item_learning_progress
                
    def predict_p_correct(self, student_id, skill):
        """
            Description - predict_p_correct()
            ---------------------------------
                Function to do predict P(correct) based on current BKT parameter values; usually called after fitting
            
            Parameters
            ----------
            student_id  --> type int    
                            index of student in transac_student_ids whose P(Correct) we need to predict

            skill       --> type int
                            index of skill in self.KC_subtest whose P(Correct) we need to predict
        """
        print("PREDICTING P(Correct)....")
        i = student_id
        j = skill
        
        p_know = self.know[i][j]
        p_not_know = 1.0 - p_know
        
        p_guess = self.guess[i][j]
        
        p_slip = self.slip[i][j]
        p_not_slip = 1.0 - p_slip

        correct = (p_know * p_not_slip) + (p_not_know * p_guess)
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
        update()              : Does the activity wise BKT update for "one observation" of an activity 
        fit()                 : Does actiity wise BKT update for some number of observations by calling the update() function
        predict_p_correct()   : After fitting, this returns P(Correct) of getting an item correct based on learnt BKT params 
    """
    def __init__(self, know, guess, slip, learn, KC_subtest, forget, num_students, num_skills):
        self.n = num_students
        self.u = num_skills
        self.timestep = np.zeros((self.n, self.u))
        self.know = know
        self.guess = guess
        self.slip = slip
        self.learn = learn
        self.forget = forget
        self.KC_subtest = KC_subtest
    
    """Function to do the bayesian update, ie calc P(Know @t+1 | obs @t+1) based on P(Know @t), guess slip etc. 
       Use this estimate along with P(Learn) to find P(Know @t+1)
    """
    def update(self, observation, i, j, activity_learning_progress, act_student_ids):

        """
            Description - update()
            ----------------------
                Function to do the activity wise bayesian update per observation. 
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

        unique_child_id = act_student_ids[i]
        
        activity_learning_progress[unique_child_id][j].append(posterior_know)

        return activity_learning_progress

    def fit(self, activity_observations, student_ids, skills, activity_learning_progress, activity_student_ids):
        """
            Description - fit()
            -------------------
                Function to do the bayesian update for k observations.
                Done by calling update() for every observation 

            Parameters
            ----------
            activity_observations:  type [float] of length k
        
            student_ids:    type [int] of length k
                            ith entry of item_observations corresponds to Unique_Child_ID_1 = activity_student_ids[student_ids[i]]

            skills:     type 2d list [[int]] of shape (k, self.u) where self.u is the total num_skills
                        all values are either 1 or 0
                        for ith activity observation, we look at skills[i] which is a list of len self.u
                        if skills[i][j] == 1,it means that activity_observations[i] is linked with skill self.KC_subtest[j]; else no

            activity_learning_progress: type dict with key as "Unique_Child_ID_1" 
                                        value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                                        shape of value: (u, *)
            
            activity_student_ids: list of unique "Unique_Child_ID_1" from activity table maintaining order of first appearance in the activity table
            
            Returns
            -------
            activity_learning_progress:  type dict
                                        Update activity_learning_progress which is done by self.update()
        """

        print("FITTING BASED ON OBSERVATIONS....")
        k = len(activity_observations)
        # For each attempt in item_observations
        for i in range(k):
            # For each skill
            for j in range(self.u):
                # if this activity exercises the skill
                if(skills[i][j] == 1):
                    # perform a BKT update
                    activity_learning_progress = self.update(activity_observations[i], student_ids[i], j, activity_learning_progress, activity_student_ids)
        
        return activity_learning_progress
    
    def predict_percent_correct(self, student_id, skill):
        print("PREDICTING P(Correct)....")
        i = student_id
        j = skill
        
        p_know = self.know[i][j]
        p_not_know = 1.0 - p_know
        
        p_guess = self.guess[i][j]
        
        p_slip = self.slip[i][j]
        p_not_slip = 1.0 - p_slip

        correct = (p_know * p_not_slip) + (p_not_know * p_guess)
        return correct

def plot_learning(learning_progress, kc_list, student_ids):
    """
        Description - plot_learning()
        -----------------------------

        Parameters
        ----------
        learning_progress:  type dict with key as "Anon Student ID" or "Unique_Child_ID_1"; could be activity_learning_progress or item_learning_progress
                            value is a 2Dlist of P(Knows) for every update of P(Know) that happened to every skill for this student
                            shape of value: (u, *)

        kc_list:    List of Skills as per CTA

        student_ids:    List of indices of students; transac_student_ids[student_id] gives "Anon Student ID" and acitivity_student_ids[student_id] gives "Unique_Child_ID_1"
        
        Prints
        ------
        Skill-wise P(Know) for each student
        
        Plots
        -----
        P(Know) vs #opportunity per student per skill
        
        Returns
        -------
        No return
    """
    for key in learning_progress:
        print("Learning progress of student ID:", key)
        r = 0
        for row in learning_progress[key]:
            skill = kc_list[r]
            while len(skill) < 25:
                skill = skill + " "
            print(skill, row)
            r += 1

    for key in learning_progress:
        # student_id = student_ids[r]
        r = 0        
        knows = learning_progress[key]
        for know in knows:
            skill = kc_list[r]
            r += 1
            if len(know) < 2:   continue
            X = np.arange(len(know)).tolist()
            plt.plot(X, know)
            plt.scatter(X, know)
            plt.title("Opportunity versus Pr(Know) for skill: " + skill + " for student: " + str(key))
            plt.xlabel("Opportunity")
            plt.ylabel("Pr(Know)")
            plt.ylim(0, 1.1)
            plt.show()

# Pandas data reading
transac_df = pd.read_csv('Data/transactions_village114.txt', sep='\t')[:100]
drop_cols = ["Row", "Sample Name", "Session Id","Time","Problem Start Time","Time Zone", "Duration (sec)", "Student Response Type", "Student Response Subtype", "Tutor Response Type", "Tutor Response Subtype", "Selection", "Action", "Feedback Text", "Feedback Classification", "Help Level", "Total Num Hints", "School", "Class", "CF (File)", "CF (Hiatus sec)", "CF (Original DS Export File)","CF (Unix Epoch)","CF (Village)","Event Type","CF (Student Used Scaffold)","CF (Robotutor Mode)","KC Category (Single-KC)","KC Category (Unique-step)","CF (Child Id)","CF (Date)","CF (Placement Test Flag)","CF (Week)","Transaction Id","Problem View","KC (Single-KC)","KC (Unique-step)", "CF (Activity Finished)", "CF (Activity Started)", "CF (Attempt Number)","CF (Duration sec)", "CF (Expected Answer)", "CF (Matrix)", "CF (Matrix Level)", "CF (Matrix Order)", "CF (Original Order)","CF (Outcome Numeric)", "CF (Placement Test User)", "CF (Problem Number)", "CF (Session Sequence)", "CF (Student Chose Repeat)", "CF (Total Activity Problems)", "CF (Tutor Sequence Session)", "CF (Tutor Sequence User)","Input", "Is Last Attempt", "Attempt At Step", "Step Name"]
transac_df = transac_df.drop(columns=drop_cols)
transac_df = transac_df.astype({"Anon Student Id": str})
item_student_ids = pd.unique(transac_df["Anon Student Id"].values.ravel()).tolist()
transac_tutor_ids = transac_df["Level (Tutor)"].values.ravel().tolist()

# has "_" instead of ":" in cta_df but might have an extra "__it_2" etc., at the end
# removing "__it_n" at the end in transac_tutor_ids
for i in range(len(transac_tutor_ids)):
    idx = transac_tutor_ids[i].find("__it_")
    if idx != -1:
        transac_tutor_ids[i] = transac_tutor_ids[i][:idx]

cta_df = pd.read_excel("Data/CTA.xlsx")
cta_columns = cta_df.columns.tolist()

# Variable initializations
activity_learning_progress = {}
item_learning_progress = {}
act_student_id_to_number_map = {}
underscore_to_colon_tutor_id_dict = {}
col_values = []
cta_tutor_ids = []
kc_to_tutorID_dict = {}
tutorID_to_kc_dict = {}
activity_observations = []
kc_list = cta_columns
skill_to_number_map = {}
student_id_to_number_map = {}
n = len(item_student_ids)
u = len(kc_list)

for column in cta_columns:
    kc_to_tutorID_dict[column] = []

for column in cta_columns:
    col_values = cta_df[[column]].values.ravel()
    remove_idx = []
    for i in range(len(col_values)):
        if(isinstance(col_values[i], str) == False):
            col_values[i] = str(col_values[i])
        if col_values[i].lower() == 'nan' or col_values[i].lower()[:6] == 'column':
            remove_idx.append(i)
    
    col_values = np.delete(col_values, remove_idx)
    kc_to_tutorID_dict[column] = col_values
    for val in col_values:
        cta_tutor_ids.append(val)
    
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

cta_tutor_ids = np.array(cta_tutor_ids)
cta_tutor_ids = pd.unique(cta_tutor_ids)

# Convert all ":" to "_"
for i in range(len(cta_tutor_ids)):
    underscored_id = cta_tutor_ids[i].replace(":", "_")
    colon_id = cta_tutor_ids[i]
    underscore_to_colon_tutor_id_dict[underscored_id] = colon_id
    cta_tutor_ids[i] = underscored_id

for ids in transac_tutor_ids:
    if ids not in cta_tutor_ids:
        print("ID MISSING IN CTA TABLE ", ids)

# Initial values for itemBKT variables/parameters. def stands for default
def_item_knew = 0.08 * np.ones((n, u))
def_item_guess = 0.22 * np.ones((n, u))
def_item_slip = 0.08 * np.ones((n, u))
def_item_learn = 0.2 * np.ones((n, u))
def_item_forget = np.zeros((n, u))

for i in range(n):
    student_id = item_student_ids[i]
    item_learning_progress[(student_id)] = np.array([def_item_knew[0]]).T.tolist()

# learning_progress[i][j][k] gives P(Knew) of student i skill j at timestep or opportunity k

item_observations = transac_df["Outcome"].values.ravel().tolist()
item_student_ids = transac_df["Anon Student Id"].values.ravel().tolist()
item_skills = [0] * len(transac_tutor_ids)

for i in range(len(item_skills)):
    underscore_tutor_id = transac_tutor_ids[i]
    colon_tutor_id = underscore_to_colon_tutor_id_dict[underscore_tutor_id]
    item_skills[i] = tutorID_to_kc_dict[colon_tutor_id]

for i in range(len(kc_list)):
    skill = kc_list[i]
    skill_to_number_map[skill] = i

for i in range(len(item_skills)):
    row = item_skills[i]
    res = [0] * len(kc_list)
    for skill in row:
        res[skill_to_number_map[skill]] = 1
    item_skills[i] = res

transac_student_ids = pd.unique(transac_df["Anon Student Id"].values.ravel()).tolist()

for i in range(len(transac_student_ids)):
    student_id = transac_student_ids[i]
    student_id_to_number_map[student_id] = i

for i in range(len(item_student_ids)):
    student_id = item_student_ids[i]
    item_student_ids[i] = student_id_to_number_map[student_id]

ibkt = ItemBKT(def_item_knew, def_item_guess, def_item_slip, def_item_learn, kc_list, def_item_forget, n, u)
item_learning_progress = ibkt.fit(item_observations, item_student_ids, item_skills, item_learning_progress, transac_student_ids)
plot_learning(item_learning_progress, kc_list, item_student_ids)












# Activity BKT
activity_df = pd.read_csv('Data/newActivityTable.csv')[:100]
activity_names = activity_df["ActivityName"].values.ravel().tolist()
student_ids = activity_df["Unique_Child_ID_1"].values.ravel().tolist()
activity_student_ids = pd.unique(activity_df["Unique_Child_ID_1"].values.ravel()).tolist()
activity_skills = [0] * len(student_ids)

for i in range(len(activity_names)):
    activity_name = activity_names[i]
    idx = activity_name.find("__it_")
    if idx != -1:
        activity_names[i] = activity_name[:idx]

for i in range(len(activity_names)):
    activity_name = activity_names[i]
    if activity_name in tutorID_to_kc_dict:
        activity_skills[i] = tutorID_to_kc_dict[activity_name]
    
    elif activity_name == "activity_selector":
        underscored_id = activity_df["TutorID"].values.ravel().tolist()[i].strip()
        idx = underscored_id.find("__it_")
        if idx != -1:
            underscored_id = underscored_id[:idx]
        colon_id = underscore_to_colon_tutor_id_dict[underscored_id]
        activity_skills[i] = tutorID_to_kc_dict[colon_id]
    
    else:
        activity_skills[i] = [0] * u
        print("ACTIVITY NAME: ", activity_name, "is NOT VALID")

for i in range(len(activity_skills)):
    row = activity_skills[i]
    res = [0] * len(kc_list)
    for skill in row:
        if skill in skill_to_number_map:
            res[skill_to_number_map[skill]] = 1
    activity_skills[i] = res

n = len(activity_student_ids)
num_corrects = activity_df["total_correct_attempts"].values.ravel().tolist()
num_attempts = activity_df["#attempts"].values.ravel().tolist()

print(num_corrects)
print(num_attempts)

for i in range(len(num_corrects)):
    percent_correct = num_corrects[i]/num_attempts[i]
    activity_observations.append(percent_correct)

# Initial values for activity BKT variables/parameters. def stands for default
def_activity_knew = 0.1 * np.ones((n, u))
def_activity_guess = 0.25 * np.ones((n, u))
def_activity_slip = 0.1 * np.ones((n, u))
def_activity_learn = 0.3 * np.ones((n, u))
def_activity_forget = np.zeros((n, u))

for i in range(n):
    student_id = activity_student_ids[i]
    activity_learning_progress[(student_id)] = np.array([def_activity_knew[0]]).T.tolist()

for i in range(len(activity_student_ids)):
    student_id = activity_student_ids[i]
    act_student_id_to_number_map[student_id] = i

for i in range(len(student_ids)):
    student_id = student_ids[i]
    student_ids[i] = act_student_id_to_number_map[student_id]

activity_bkt = ActivityBKT(def_activity_knew, def_activity_guess, def_activity_slip, def_activity_learn, kc_list, def_activity_forget, n, u)
activity_learning_progress = activity_bkt.fit(activity_observations, student_ids, activity_skills, activity_learning_progress, activity_student_ids)
plot_learning(activity_learning_progress, kc_list, student_ids)