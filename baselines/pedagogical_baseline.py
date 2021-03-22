from simulators.student_simulator import StudentSimulator
from simulators.tutor_simulator import TutorSimulator
from environment import StudentEnv

from constants import CONSTANTS

from tqdm import tqdm
import numpy as np

class PedagogicalBaseline():
    def __init__(self, CONSTANTS):
        self.student_simulator = StudentSimulator(village=CONSTANTS['VILLAGE'], observations=CONSTANTS['NUM_OBS'], student_model_name=CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=CONSTANTS['NEW_STUDENT_PARAMS'], prints=False)
        self.tutor_simulator = TutorSimulator(CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], CONSTANTS['MID_PERFORMANCE_THRESHOLD'], CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], CONSTANTS['AREA_ROTATION'], type=1, thresholds=True)
        self.CONSTANTS = CONSTANTS

    def run(self, avg_over_runs, agent_type, student_params=None, prints=False):
        performances_ys = []
        if student_params is None:  student_params = 'new_student'
        student_num = self.student_simulator.uniq_student_ids.index(student_params)
        uniq_activities = self.student_simulator.uniq_activities
        student_model_name = self.CONSTANTS['STUDENT_MODEL_NAME']

        if student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
            prior_know = np.array(self.student_simulator.student_model.alpha[student_num][-1])
            prior_avg_know = np.mean(prior_know)

        for _ in tqdm(range(avg_over_runs), desc='Pedagogical expert baseline'):
            student_simulator = StudentSimulator(village=self.CONSTANTS['VILLAGE'], observations=self.CONSTANTS['NUM_OBS'], student_model_name=self.CONSTANTS['STUDENT_MODEL_NAME'], new_student_params=student_params, prints=False)
            tutor_simulator = TutorSimulator(self.CONSTANTS['LOW_PERFORMANCE_THRESHOLD'], self.CONSTANTS['MID_PERFORMANCE_THRESHOLD'], self.CONSTANTS['HIGH_PERFORMANCE_THRESHOLD'], self.CONSTANTS['AREA_ROTATION'], type=1, thresholds=True)
            
            activity_num = None
            response = ""
            ys = []
            ys.append(prior_avg_know)
    
            for _ in range(self.CONSTANTS['MAX_TIMESTEPS']):
                if activity_num != None:
                    p_know_activity = student_simulator.student_model.get_p_know_activity(student_num, activity_num)
                else:
                    p_know_activity = None
                x, y, area, activity_name = tutor_simulator.get_next_activity(p_know_prev_activity=p_know_activity, prev_activity_num=activity_num, response=str(response), prints=prints)
                activity_num = uniq_activities.index(activity_name)
                response = student_simulator.student_model.predict_response(activity_num, student_num, update=True)
                if prints:
                    print('ASK QUESTION:', activity_num, uniq_activities[activity_num])
                    print('CURRENT MATRIX POSN (' + str(area) + '): ' + "[" + str(x) + ", " + str(y) + "]")
                    print('CURRENT AREA: ' + area)
                    print()
                if student_model_name == 'hotDINA_skill' or student_model_name == 'hotDINA_full':
                    posterior_know = np.array(student_simulator.student_model.alpha[student_num][-1])
                    posterior_avg_know = np.mean(posterior_know)
                ys.append(posterior_avg_know)
            
            performances_ys.append(ys)
        
        mean_performance_ys = np.mean(performances_ys, axis=0)
        std_performance_ys = np.std(performances_ys, axis=0)

        return mean_performance_ys, mean_performance_ys-std_performance_ys, mean_performance_ys+std_performance_ys