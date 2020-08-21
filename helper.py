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

def skill_probas(activityName, tutorID_to_kc_dict, kc_list, p_know):
    """
        Given:
            ActivityName, TutorID to KC skill mapping
        Returns:
            P_know of those skills exercised by Activity Name
    """
    # returns list of P(Know) for only those skills that are exercised by activityName
    proba = []
    
    skillNames = tutorID_to_kc_dict[activityName]
    skillNums = []
    for skillName in skillNames:
        skill_num = kc_list.index(skillName)
        skillNums.append(skill_num)
    
    for skill in skillNums:
        proba.append(p_know[skill])
    
    return proba

def clear_files(algo, clear, path=''):
    """
        Empties all txt files under algo + "_logs" folder if clear is set to True
        
        Returns nothing
    """
    if clear == False:
        return
    log_folder_name = path + algo + "_logs"

    files = os.listdir(log_folder_name)
    text_files = []
    
    for file in files:
        if file[-3:] == "txt":
            text_files.append(file)
    
    for text_file in text_files:
        file = open(log_folder_name + "/" + text_file, "r+")
        file.truncate(0)
        file.close()

def output_avg_p_know(episode_num, avg_over_episodes, scores, filename, avg_p_know):
    if episode_num % avg_over_episodes == 0 and episode_num > 0:
        avg_score = np.mean(scores[max(0, episode_num-avg_over_episodes): episode_num+1])
        with open(filename, "a") as f:
            text = str(episode_num/avg_over_episodes) + ", Avg P(Know)" + str(avg_p_know) + ", Avg score:" + str(avg_score) + "\n"
            f.write(text)

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
        plt.plot(x, y, label=student_id)
    
    x = np.arange(1, len(new_student_avg) + 1).tolist()
    plt.plot(x, new_student_avg, label="RL Agent", color="black")
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
    
    theta   =  []
    lambda0 =  []
    lambda1 =  []
    learn   =  []
    g       =  []
    ss      =  []
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

def evaluate_performance_thresholds(student_simulator, tutor_simulator, CONSTANTS=None, prints=True):

    student_id = CONSTANTS['STUDENT_ID']
    student_num = student_simulator.uniq_student_ids.index(student_id)
    uniq_activities = student_simulator.uniq_activities
    student_model_name = CONSTANTS['STUDENT_MODEL_NAME']
    
    if CONSTANTS["STUDENT_MODEL_NAME"] == 'hotDINA_skill' or CONSTANTS['STUDENT_MODEL_NAME'] == 'hotDINA_full':
        prior_know = np.array(student_simulator.student_model.alpha[student_num][-1])
        prior_avg_know = np.mean(prior_know)
    
    if prints: print()
    activity_num = None
    response = ""
    ys = []
    ys.append(prior_avg_know)
    for _ in range(CONSTANTS['MAX_TIMESTEPS']):
        if activity_num != None:
            p_know_activity = student_simulator.student_model.get_p_know_activity(student_num, activity_num)
        else:
            p_know_activity = None
        x, y, area, activity_name = tutor_simulator.get_next_activity(p_know_activity, activity_num, str(response), prints)
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
    return ys
    


