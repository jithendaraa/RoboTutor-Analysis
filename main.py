import os
from helper import *
from simulators.simulator import Simulator
from constants import CONSTANTS

if __name__ == '__main__':  
    simulator = Simulator()
    CONSTANTS = simulator.set_constants(CONSTANTS)
    simulator.set_data_paths(CONSTANTS, root_path=os.getcwd())
    # clear_files(CONSTANTS['ALGO'], CONSTANTS['CLEAR_LOGS'])
    # simulator.evaluate_current_RT_thresholds(CONSTANTS, prints=False, avg_over_runs=30)
    # CONSTANTS = simulator.learn(CONSTANTS)
    simulator.play(CONSTANTS)