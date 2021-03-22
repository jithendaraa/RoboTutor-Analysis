"""
    1. Download all village transactions data (with KC) from 
        https://drive.google.com/drive/folders/1tjj2jh3-PRroN-PBxNT05A86Pyub6ZhB
        into same folder as this scripts file
        These villages are named analogous to "village_114_KCSubtests.txt" and village numbers range from 114 to 141
    2. `git clone https://github.com/myudelson/hmm-scalable`
        rename hmm-scalable to hmmscalable
        `cd hmmscalable`
        `make all`
        
    Params are Knew Learn Slip Guess in this order
"""

import os 
import numpy as np
import pandas as pd

default_guess_value = "0.3"
default_slip_value = "0.2"

# Villages for which data needs to be fit: (114, 142) means data will be fit for village 114 to 141
village_nums = np.arange(114, 142).tolist()
for i in range(len(village_nums)):
    village_nums[i] = str(village_nums[i])

paths_to_transactions_kc_tables = []

for village in village_nums:
    # path to transactions file for village `village` where `village` is a string from 114 to 141   
    path_to_transactions_file = "village_" + village + "_KCSubtests.txt"
    paths_to_transactions_kc_tables.append(path_to_transactions_file)


# Create a folder village_n and move "village_n_KCSubtests.txt" into village_n 
for i in range(len(village_nums)):
    village = village_nums[i]
    os.system("mkdir village_" + village)
    os.system("mv village_" + village + "_KCSubtests.txt village_" + village + "/village_" + village + "_KCSubtests.txt")

"""
    Using village_n/village_n_KCSubtests.txt create: 
        1.   village_n/village_n_step_transac.txt               :   extracts data from village_n_KCSubtests.txt into the format required by hmmscalable tool
        2.   village_n/village_n_step_transac_single_student.txt:   extracts data from village_n_KCSubtests.txt treating the data 
                                                                    related to one student with id 'student'. Useful as this will 
                                                                    be the data to fit BKT params assuming there is only one student but 22 KC skills.
                                                                    Data format is compatible with hmmscalable tool
"""

for village in village_nums:
    path_to_village_data = "village_"+ village +"/village_" + village + "_KCSubtests.txt"
    village_df = pd.read_csv(path_to_village_data, delimiter='\t')
    student_ids = village_df["Anon Student Id"].values.ravel()
    problem_names = village_df['Problem Name'].values.tolist()
    step_names = village_df['Step Name'].values.tolist()
    problem_steps = []
    
    for i in range(len(problem_names)):
        problem_steps.append("problem-" + problem_names[i].replace(" ", "") + "-step-" + str(step_names[i]))
    problem_steps = np.array(problem_steps)

    responses = village_df["Outcome"].values.tolist()
    for i in range(len(responses)):
        response = responses[i]
        if response == "CORRECT":
            responses[i] = 1
        else:
            responses[i] = 2
    responses = np.array(responses)
    
    kc_0 = village_df['KC(Subtest)'].values.tolist()
    kc_1 = village_df['KC(Subtest).1'].values.tolist()
    kc_2 = village_df['KC(Subtest).2'].values.tolist()
    kc_3 = village_df['KC(Subtest).3'].values.tolist()
    skills = []

    for i in range(len(kc_0)):
        skill = ""
        
        if isinstance(kc_0[i], str) == False:
            print(kc_0[i], "Error in File: ", village)
        else:
            skill = kc_0[i].replace(" ", "")
        if isinstance(kc_1[i], str) == True:
            skill = skill + "~" + kc_1[i].replace(" ", "")
        if isinstance(kc_2[i], str) == True:
            skill = skill + "~" + kc_2[i].replace(" ", "")
        if isinstance(kc_3[i], str) == True:
            skill = skill + "~" + kc_3[i].replace(" ", "")

        if skill == "":
            skill = "."

        skills.append(skill)
    
    single_students = np.array(['student'] * len(skills))
    skills = np.array(skills)
    step_transaction = np.vstack((responses, student_ids, problem_steps, skills)).T
    step_transaction_df = pd.DataFrame(step_transaction)
    step_transaction_df.to_csv("village_" + village+ "/village_" + village + "_step_transac.txt", index=None, header=None, sep='\t')
    
    single_student_transaction = np.vstack((responses, single_students, problem_steps, skills)).T
    single_student_step_transaction_df = pd.DataFrame(single_student_transaction)
    single_student_step_transaction_df.to_csv("village_" + village+ "/village_" + village + "_step_transac_single_student.txt", index=None, header=None, sep='\t')
    print("Parsed data from village " + village)

"""
    Fit BKT Params for every village assuming all data is based on a single student.
    Result is village_n/village_n_single_student_model.txt
"""
for village in village_nums:
    
    transac_data_path = "../village_" + village + "/village_" + village + "_step_transac_single_student.txt"
    model_output_file_path = "../village_" + village + "/village_" + village + "_single_student_model.txt"
    fit_bkt_command = "./trainhmm -s 1.1 -d ~ -m 1 -p 1 " + transac_data_path + " " + model_output_file_path + " predict.txt"
    
    os.chdir('hmmscalable')
    os.system(fit_bkt_command)
    os.chdir('..')

path_to_files = []
    
for village in village_nums:
    path_to_files.append("village_" + village + "/village_" + village + "_step_transac.txt")

uniq_student_ids = []
student_id_to_village_map = {}

"""
    Student wise BKT Fit models for each village. Student specific data is first generated as studentID_here.txt 
    in the "village_n" directory where n is the village to which the student "studentID" belongs.
    Model outputted by hmmscalable is a txt file named studentID_model.txt which contains fitted params for studentID
"""

for i in range(len(path_to_files)):
    file_path = path_to_files[i]
    village = village_nums[i]
    file = open(file_path, "r").read()
    lines = file.split("\n")
    for j in range(len(lines)):
        if j == len(lines) - 1:
            break
        line = lines[j].split("\t")
        student_id = line[1]
        if student_id not in uniq_student_ids:
            uniq_student_ids.append(student_id)
            student_id_to_village_map[student_id] = [village, j]

for student_id in uniq_student_ids:
    village = student_id_to_village_map[student_id][0]
    idx = student_id_to_village_map[student_id][1]
    
    file_path = "village_" + village + "/village_" + village + "_step_transac.txt"
    file = open(file_path, "r")
    lines = file.read().split("\n")
    n = len(student_id)
    output_lines = ""
    start = 0
    for line in lines:
        if len(line) < n:
            break
        if line.split("\t")[1] == student_id:
            output_lines += line + "\n"
            start = 1
        elif start == 1:
            start = -1
            break
    output_lines = output_lines[:-1]
    file.close()
    
    data_file = "village_" + village + "/" + str(student_id) +".txt"
    file = open(data_file, "w")
    file.write(output_lines)
    file.close()
    
    transac_data_path = "../village_" + village + "/" + str(student_id) +".txt"
    model_output_file_path = "../village_" + village + "/" + str(student_id) +"_model.txt"
    fit_bkt_command = "./trainhmm -s 1.1 -d ~ -m 1 -p 1 " + transac_data_path + " " + model_output_file_path + " predict.txt"
    
    os.chdir('hmmscalable')
    os.system(fit_bkt_command)
    os.chdir('..')

for village in village_nums:
    single_student_data = "village_" + village + "/village_" + village + "_step_transac_single_student.txt"
    file = open(single_student_data, "r")
    contents = file.read().split("\n")[:-1]
    lines = ""
    for line in contents:
        vals = line.split('\t')
        vals[len(vals) - 1] = "skill_1"
        
        lines += vals[0] + '\t' + vals[1] + '\t' + vals[2] + '\t' + 'skill_1' + '\n'
    
    file.close()
    
    output_file_name = "village_" + village + "/village_"+village+"_step_transac_single_student_single_skill.txt"
    output_file = open(output_file_name, "w")
    output_file.writelines(lines)
    output_file.close()

single_student_single_skill_data_paths = []

for village in village_nums:
    path = "../village_" + village + "/village_" + village + "_step_transac_single_student_single_skill.txt"
    single_student_single_skill_data_paths.append(path)
    
    model_output_file_path = "../village_" + village + "/village_" + village + "_single_student_single_skill_model.txt"
    fit_bkt_command = "./trainhmm -s 1.1 -d ~ -m 1 -p 1 " + path + " " + model_output_file_path + " predict.txt"
    
    os.chdir('hmmscalable')
    os.system(fit_bkt_command)
    os.chdir('..')



"""
    Convert all studentID_model.txt files into studentID_params.txt and save it in village_n/params
    studentID_model.txt is inconvenient to read, you might prefer to read studentID_params.txt instead.
    One line of studentID_params.txt contains skill_name knew learn slip guess and is easier to make sense of than a studentID_model.txt file
"""

for village in village_nums:
    village_data_path = "village_" + village
    os.chdir(village_data_path)
    os.system("mkdir params")
    files = os.listdir('.')
    model_files = []
    for file in files:
        if len(files) >= 9 and file[-9:] == 'model.txt':
            model_files.append(file)
    for model_file_name in model_files:
        model_file = open(model_file_name, "r")
        file_lines = model_file.read().split('\n')[7:-1]
        lines = ""
        for i in range(len(file_lines)):
            if i == len(file_lines) - 1:
                break
            file_lines[i] = file_lines[i].split('\t')
            if file_lines[i][0].isnumeric() == True:
                guess = default_guess_value
                if len(file_lines[i+3].split('\t')) >= 4:
                    guess = file_lines[i+3].split('\t')[3]     
                lines += file_lines[i][1] + "\t" + str(file_lines[i+1].split('\t')[1]) + "\t" + str(file_lines[i+2].split('\t')[3]) + "\t" + str(file_lines[i+3].split('\t')[2]) + "\t" + guess + "\n" 
        model_file.close()
        output_file = open("params/" + model_file_name[:-9] + "params.txt", "w")
        output_file.write(lines)
        output_file.close()
    os.chdir('..')

# Move all studentID_model.txt into model/

for village in village_nums:
    os.chdir('village_' + village)
    os.system("mkdir model")
    files = os.listdir('.')
    model_files = []
    for file in files:
        if len(files) >= 9 and file[-9:] == 'model.txt':
            model_files.append(file)
    for file_path in model_files:
        move_into_model_directory_cmd = "mv " + file_path + " model/" + file_path
        os.system(move_into_model_directory_cmd)
    os.chdir('..')