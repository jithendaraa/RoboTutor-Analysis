import torch
import argparse
from constants import CONSTANTS

class Simulator():
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu:0")
        print('Device:', self.device)
        self.args = self.arg_parser()
        self.algo = self.args.algo

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
        parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
        parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
        parser.add_argument("-e", "--num_envs", default=CONSTANTS["NUM_ENVS"], help="Number of observations to train on", type=int)
        parser.add_argument('-mt', '--max_timesteps', help="Total questions that will be given to the student/RL agent", type=int, default=CONSTANTS['MAX_TIMESTEPS'])
        parser.add_argument('-sid', '--student_id', default=CONSTANTS['STUDENT_ID'], help="Student id")
        parser.add_argument('-smn', '--student_model_name', default=CONSTANTS['STUDENT_MODEL_NAME'], help="Student model name")
        parser.add_argument('--test_epochs', default=CONSTANTS['TEST_EPOCHS'], help="Test after every x epochs", type=int)
        parser.add_argument('--ppo_steps', help="PPO Steps", default=CONSTANTS['PPO_STEPS'], type=int)
        parser.add_argument('-ar', '--area_rotation', help="Area rotation sequence like L-N-L-S", default=CONSTANTS['AREA_ROTATION'])
        parser.add_argument('-arc', '--area_rotation_constraint', help="Should questions be constrained like lit-num-lit-stories? True/False", default=True)
        parser.add_argument('-tc', '--transition_constraint', help="Should transition be constrained to prev,same,next, next-next? True/False", default=True)
        parser.add_argument('--target', default=CONSTANTS['TARGET_P_KNOW'], type=float)
        parser.add_argument('--anti_rl', default=False)
        parser.add_argument('-m', '--model', help="Model file to load from checkpoints directory, if any")
        parser.add_argument('-nsp', '--new_student_params', help="The model params new_student has to start with; enter student_id")
        parser.add_argument("-c", "--clear_logs", default=CONSTANTS["CLEAR_LOGS"])
        parser.add_argument("--algo", default=CONSTANTS["ALGO"], help="DQN/A3C/PPO")
        parser.add_argument("--avg_runs", default=CONSTANTS["AVG_OVER_RUNS"], help="Number of runs to average experiments")
        args = parser.parse_args()
        return args

    def set_constants(self, CONSTANTS):
        args = self.args
        CONSTANTS['NUM_OBS']                    = args.observations
        CONSTANTS['VILLAGE']                    = args.village_num
        CONSTANTS['AGENT_TYPE']                 = args.type
        CONSTANTS['NUM_ENVS']                   = args.num_envs
        CONSTANTS['MAX_TIMESTEPS']              = args.max_timesteps
        CONSTANTS['STUDENT_ID']                 = args.student_id
        CONSTANTS['STUDENT_MODEL_NAME']         = args.student_model_name
        CONSTANTS['TEST_EPOCHS']                = args.test_epochs
        CONSTANTS['PPO_STEPS']                  = args.ppo_steps
        CONSTANTS['AREA_ROTATION']              = args.area_rotation
        CONSTANTS['AREA_ROTATION_CONSTRAINT']   = args.area_rotation_constraint
        CONSTANTS['TRANSITION_CONSTRAINT']      = args.transition_constraint
        CONSTANTS['TARGET_P_KNOW']              = args.target
        CONSTANTS['NEW_STUDENT_PARAMS']         = args.new_student_params
        CONSTANTS['CLEAR_LOGS']                 = args.clear_logs
        CONSTANTS['AVG_OVER_RUNS']              = args.avg_runs

        if args.type == None:
            CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS']
            CONSTANTS['USES_THRESHOLDS'] = None
            CONSTANTS['TRANSITION_CONSTRAINT'] = False
            CONSTANTS['AREA_ROTATION_CONSTRAINT'] = False

        if args.type == 1:  # State size: number of KC's; Action size: 3 threshold values
            CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS']
            CONSTANTS['ACTION_SIZE'] = 3
            CONSTANTS['USES_THRESHOLDS'] = True
            CONSTANTS['FC1_DIMS'] = 256

        elif args.type == 2:    # State size: number of KC's + 1 matrix_type state + 1 position state; Action size: 3 threshold values
            CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS'] + 1 + 1
            CONSTANTS['ACTION_SIZE'] = 3
            CONSTANTS['USES_THRESHOLDS'] = True
            CONSTANTS['FC1_DIMS'] = 128
            CONSTANTS['FC2_DIMS'] = 256

        elif args.type == 3:
            CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS'] + 1 + 1
            CONSTANTS['ACTION_SIZE'] = 4    # prev, same, next, next_next
            CONSTANTS['USES_THRESHOLDS'] = False
            CONSTANTS['FC1_DIMS'] = 128
            CONSTANTS['FC2_DIMS'] = 256
            CONSTANTS['LEARNING_RATE'] = 1e-4

        elif args.type == 4:
            CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS'] + 1 
            CONSTANTS['ACTION_SIZE'] = None
            CONSTANTS['USES_THRESHOLDS'] = False
            CONSTANTS['TRANSITION_CONSTRAINT'] = False
            CONSTANTS['AREA_ROTATION_CONSTRAINT'] = True
            CONSTANTS['FC1_DIMS'] = 128
            CONSTANTS['FC2_DIMS'] = 256
            CONSTANTS['LEARNING_RATE'] = 5e-3

        elif args.type == 5:
            CONSTANTS['STATE_SIZE'] = CONSTANTS['NUM_SKILLS']
            CONSTANTS['ACTION_SIZE'] = None
            CONSTANTS['USES_THRESHOLDS'] = False
            CONSTANTS['TRANSITION_CONSTRAINT'] = False
            CONSTANTS['AREA_ROTATION_CONSTRAINT'] = False
            CONSTANTS['FC1_DIMS'] = 128
            CONSTANTS['FC2_DIMS'] = 256
            CONSTANTS['LEARNING_RATE'] = 5e-3

        return CONSTANTS
