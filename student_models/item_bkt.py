import numpy as np

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
    def __init__(self, params_dict, kc_list, num_students, uniq_student_ids):
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

        self.know = params_dict['know']
        self.guess = params_dict['guess']
        self.slip = params_dict['slip']
        self.learn = params_dict['learn']
        self.forget = params_dict['know']
        
        self.kc_list = kc_list
        self.n = num_students
        self.u = len(kc_list)
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