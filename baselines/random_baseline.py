from simulators.student_simulator import StudentSimulator
from simulators.tutor_simulator import TutorSimulator
from environment import StudentEnv

from constants import CONSTANTS

from tqdm import tqdm
import numpy as np

class RandomBaseline():
    def __init__(self, CONSTANTS, student_id):
        self.student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=student_id, prints=False)
        self.tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], CONSTANTS['AREA_ROTATION'], type=1, thresholds=True)
        self.CONSTANTS = CONSTANTS

    def run(self, avg_over_runs, agent_type, student_id=None):

        learning_progresses = []
        if student_id == None:  student_id = 'new_student'
        student_num = self.student_simulator.uniq_student_ids.index(student_id)
        uniq_activities = self.student_simulator.uniq_activities
        student_model_name = self.CONSTANTS['STUDENT_MODEL_NAME']
        posteriors = []

        if student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
            prior_know = np.array(self.student_simulator.student_model.alpha[student_num][-1])
            prior_avg_know = np.mean(prior_know)
        
        env = StudentEnv(self.student_simulator, self.CONSTANTS["ACTION_SIZE"], student_id, self.CONSTANTS['NUM_ENVS'], agent_type, prints=False, area_rotation=self.CONSTANTS['AREA_ROTATION'], CONSTANTS=self.CONSTANTS, anti_rl=False)
        env.checkpoint()

        for _ in tqdm(range(avg_over_runs), desc='Random baseline'):
            score, timesteps = 0, 0
            done = False
            state = env.reset()

            while done is False:
                timesteps += 1
                CONSTANTS['RUN'] += 1
                action = env.choose_random_action(CONSTANTS['AGENT_TYPE'])
                prior_know = state[:CONSTANTS['NUM_SKILLS']]
                next_state, reward, student_response, done, posterior_know = env.step(action, CONSTANTS["MAX_TIMESTEPS"], timesteps, reset_after_done=False)
                score += reward
                state = next_state.copy()
                gain = np.mean(np.array(posterior_know) - np.array(prior_know)) # mean gain in prior -> posterior knowledge of student

            posterior = state[:CONSTANTS['NUM_SKILLS']]
            posteriors.append(posterior)

            if student_model_name == 'hotDINA_full' or student_model_name == 'hotDINA_skill':
                learning_progress = env.student_simulator.student_model.alpha
                learning_progresses.append(learning_progress[student_num])
        
        learning_progresses = np.array(learning_progresses)
        posteriors = np.mean(posteriors, axis=0)
        avg_learning_progresses = np.mean(learning_progresses, axis=2)
        
        learning_progress_means = np.mean(avg_learning_progresses, axis=0)[:]
        learning_progress_stds = np.std(avg_learning_progresses, axis=0)[:]
        learning_progress_min = learning_progress_means - learning_progress_stds
        learning_progress_max = learning_progress_means + learning_progress_stds

        return learning_progress_means, learning_progress_min, learning_progress_max