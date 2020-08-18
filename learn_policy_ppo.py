import os
import argparse
import numpy as np
from pathlib import Path
import pickle

from student_simulator import StudentSimulator

from reader import *

CONSTANTS = {
    'NUM_OBS'           :   '100',
    'VILLAGE'           :   '130',
    'STUDENT_MODEL_NAME':   'hotDINA_skill',
    'AGENT_TYPE'        :   1
}

def set_constants(args):
    CONSTANTS['NUM_OBS'] = args.observations
    CONSTANTS['VILLAGE'] = args.village_num
    CONSTANTS['STUDENT_MODEL_NAME'] = args.student_model_name
    CONSTANTS['AGENT_TYPE'] = args.type

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observations", default=CONSTANTS["NUM_OBS"], help="Number of observations to train on")
    parser.add_argument("-v", "--village_num", default=CONSTANTS["VILLAGE"], help="Village to train on (not applicable for Activity BKT)")
    parser.add_argument('-t', '--type', help="RL Agent type (1-5)", type=int)
    parser.add_argument('-smn', '--student_model_name', help="Student model name")
    args = parser.parse_args()
    set_constants(args)

    kc_list, kc_to_tutorID_dict, tutorID_to_kc_dict, cta_tutor_ids, uniq_skill_groups, skill_group_to_activity_map  = read_data()
    student_simulator = StudentSimulator(village=CONSTANTS["VILLAGE"], 
                                        observations=CONSTANTS["NUM_OBS"], 
                                        student_model_name=CONSTANTS["STUDENT_MODEL_NAME"])
    