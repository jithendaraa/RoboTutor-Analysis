import numpy as np

# The Environment that the RL agent interacts with
# Given an action returns the new_state, reward, student_response, done and posterior know (P(Know) after student attempt)

class AbsSpace():
    def __init__(self, size):
        self.shape = (size, )
        
class Discrete():
    def __init__(self, size):
        self.shape = (size, )

class StudentEnv():
    def __init__(self, initial_state, activities, activity_bkt, tutorID_to_kc_dict, student_id, skill_to_number_map, skill_groups, skill_group_to_activity_map, action_size, env_num=1):
        """
            initial_state: numpy ndarray with shape (state_space, )
            activities: list of all activities/tutors from CTA Table
            state_size: #state variables in the environment; currently 18 ie, 1 state variable for each P(Know) per skill
            action_size: 1710, ie., total number of unique tutors/activities from CTA table but is bucketed into 33 "skill_groups" for now
        """
        self.initial_state                  = initial_state
        self.activities                     = activities
        self.activity_bkt                   = activity_bkt
        self.activity_to_skills_map         = tutorID_to_kc_dict
        self.student_id                     = student_id
        self.skill_to_number_map            = skill_to_number_map
        self.skill_groups                   = skill_groups
        self.skill_group_to_activity_map    = skill_group_to_activity_map
        self.action_size                    = action_size
        self.state                          = initial_state.copy()
        self.state_size                     = self.state.shape[0]
        self.observation_space              = AbsSpace(self.state_size)
        self.action_space                   = Discrete(self.action_size)
        
        if env_num is not None:
            print("Initialised RoboTutor environment number", env_num, "with", self.state_size, "states and", self.action_size, "actions")
    
    def reset(self):
        self.activity_bkt.know[self.student_id] = self.initial_state.tolist().copy()
        return self.initial_state
    
    def step(self, action, timesteps, max_timesteps):
        """
            action: int, activities[action] refers to the activity that the RL policy suggests/outputs.
        """
        done = False

        student_ids = [self.student_id]
        skills = []

        skill_group = self.skill_groups[action]
        skills = [skill_group]
        # Simulate student with current BKT params and get student response (predictions) according to full responsibility and blame-worst responsibility
        correct_preds, min_correct_preds = self.activity_bkt.predict_percent_correct(student_ids, skills)
        
        # should be min_correct_preds for "blame worst" responsibility
        student_response = correct_preds 
        
        # P(Know) as a list before attempting the question (before doing BKT update based on student_response)
        prior_know = self.activity_bkt.know[self.student_id].copy()

        # BKT update based on student_response aka %correct
        self.activity_bkt.update(activity_observations=student_response, student_ids=student_ids, skills=skills)

        # Get next state as updated P(Know) based on student_response. Set next_state as current_state
        next_state = self.activity_bkt.know[self.student_id]
        next_state = np.array(next_state)
        self.state = next_state.copy()

        # Get posterior P(Know) ie P(Know) after BKT update from student_response 
        posterior_know = self.activity_bkt.know[self.student_id].copy()

        # Get avg P(Know) before and after student attempts the activities
        avg_prior_know = np.mean(np.array(prior_know))
        avg_posterior_know = np.mean(np.array(posterior_know))

        # Reward after each attempt/opportunity
        reward = 1000 * (avg_posterior_know - avg_prior_know) # reward = 100 * np.mean(np.array(self.activity_bkt.know))

        if timesteps >= max_timesteps:
            done = True

        return next_state, reward, student_response[0], done, posterior_know