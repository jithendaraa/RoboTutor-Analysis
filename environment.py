import numpy as np

# The Environment that the RL agent interacts with
# Given an action returns the new state

class StudentEnv():
    def __init__(self, initial_state, activities, activity_bkt, tutorID_to_kc_dict, student_id, skill_to_number_map, skill_groups, skill_group_to_activity_map, state_size, action_size):
        """
            initial_state: numpy ndarray with shape (state_space, )
            activities: list of all activities/tutors from CTA Table
            state_size: #state variables in the environment; for a given student it can be 72 (4BKT params * 18 skills) or 73 (if student_level after placement/promotion is included) or could be 21(18+1+1+1)
            action_size: 1710, ie., total number of unique tutors/activities from CTA table
        """
        print("INIT ENVIRONMENT with ", state_size, " states and ", action_size, " actions")
        self.initial_state = initial_state
        self.state_dim = self.initial_state.shape
        self.state_size = state_size
        self.action_size = action_size
        self.state = initial_state.copy()
        self.activities = activities
        self.activity_bkt = activity_bkt
        self.student_id = student_id
        self.activity_to_skills_map = tutorID_to_kc_dict
        self.skill_to_number_map = skill_to_number_map
        self.skill_groups = skill_groups
        self.skill_group_to_activity_map = skill_group_to_activity_map
    
    def reset(self):
        self.activity_bkt.know[self.student_id] = self.initial_state.tolist().copy()
        return self.initial_state
    
    def step(self, action, timesteps, max_timesteps):
        """
            action: int, activities[action] refers to the activity that the RL policy suggests/outputs.
        """
        done = False
        # activity = self.activities[action]

        student_ids = [self.student_id]
        skills = []

        # skill = []
        # for row in self.activity_to_skills_map[activity]:
        #     skill.append(self.skill_to_number_map[row])
        skill_group = self.skill_groups[action]
        skills = [skill_group]

        correct_preds, min_correct_preds = self.activity_bkt.predict_percent_correct(student_ids, skills)
        
        student_response = correct_preds # could be min_correct_preds too for "blame worst" responsibility
        
        prior_know = self.activity_bkt.know[self.student_id].copy()

        # BKT update based on his response
        self.activity_bkt.update(activity_observations=student_response, student_ids=student_ids, skills=skills)
        
        next_state = self.activity_bkt.know[self.student_id]
        next_state = np.array(next_state)

        posterior_know = self.activity_bkt.know[self.student_id].copy()
        
        avg_prior_know = np.mean(np.array(prior_know))
        avg_posterior_know = np.mean(np.array(posterior_know))
        # Average P(Know) is the reward
        # reward = 100 * np.mean(np.array(self.activity_bkt.know))
        reward = 1000 * (avg_posterior_know - avg_prior_know)
        if timesteps >= max_timesteps:
            done = True
        if reward <= 0.0:
            done = True

        return next_state, reward, student_response[0], done
