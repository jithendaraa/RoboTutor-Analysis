import numpy as np
import math
import matplotlib.pyplot as plt
import os

def remove_spaces(elements):
    """
        Given 
            A list elements = ["Orienting to Print", "Listening Comprehension"]
        Returns
            elements = ["OrientingtoPrint", "ListeningComprehension"]
    """
    for i in range(len(elements)):
        elements[i] = (elements[i].replace(" ", ""))
    
    return elements

def rmse(a, b):
    """
        Calculates Root Mean Square Error between 2 arrays or lists
    """
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    
    squared_error = (a - b)**2
    mse = np.mean(squared_error)
    rmse_val = mse**0.5
    return rmse_val

def sigmoid(x):
    """
        Given
            x: int or float
        Returns
            The sigmoid (or inverse logit) of x = 1/1+e^(-x)
    """
    return 1/(1+np.exp(-x))

def remove_iter_suffix(tutor):
    """
        Given: 
            A string tutor = "something__it_(number)" or list of tutor
        Returns:    
            String tutor = "something" or list of tutor
    """
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

def skill_probas(activityName, tutorID_to_kc_dict, skill_to_number_map, p_know):
    """
        Given:
            ActivityName, TutorID to KC skill mapping, skill_to_number_map and p_know for each skill
        Returns:
            P_know of those skills exercised by Activity Name
    """
    # returns list of P(Know) for only those skills that are exercised by activityName
    proba = []
    
    skillNames = tutorID_to_kc_dict[activityName]
    skillNums = []
    for skillName in skillNames:
        skillNums.append(skill_to_number_map[skillName])
    
    for skill in skillNums:
        proba.append(p_know[skill])
    
    return proba

def clear_files(algo, clear):
    """
        Empties all txt files under algo + "_logs" folder if clear is set to True
        
        Returns nothing
    """
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

def plot_learning(learning_progress, student_ids, timesteps, new_student_avg, algo):
    """
        Given 
            learning_progress, student_ids etc.,
        Plots:
            Avg. P Know vs #opportunities for each student in student_ids.
            Also plots in the same graph, Avg P Know vs #opportunities for an imaginary student 
            who is asked questions from the learned tutor policy of the RL agent
    """

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

def slurm_output_params(path, village="130", slurm_id="10301619"):

    slurm_filename = "slurm-" + slurm_id + ".txt"
    path_to_slurm_file = path + "/" + slurm_filename
    slurm_file_lines = open(path_to_slurm_file, "rb").read().decode("utf-8").split("\n")
    theta   = {}
    lambda0 = {}
    lambda1 = {}
    learn   = {}
    g       = {}
    ss      = {}
    for line in slurm_file_lines:
        line_vals = line.split(" ")
        line_vals = list(filter(("").__ne__, line_vals))
        param_name = line_vals[0]
        if len(param_name) > 5:
            if param_name[:5] == "theta":
                student_num = len(theta)
                
                theta.append(float(line_vals[1]))
            elif param_name[:5] == "learn":
                learn.append(float(line_vals[1]))
            elif param_name[:7] == "lambda0":
                lambda0.append(float(line_vals[1]))
            elif param_name[:7] == "lambda1":
                lambda1.append(float(line_vals[1]))
        if param_name[:1] == "g":
            g.append(float(line_vals[1]))
        if param_name[:2] == "ss":
            ss.append(float(line_vals[1]))

    slurm_params_dict = {
        'theta':    theta,
        'learn':    learn,
        'b':        lambda0,
        'a':        lambda1,
        'g':        g,
        'ss':       ss
    }

    return slurm_params_dict
