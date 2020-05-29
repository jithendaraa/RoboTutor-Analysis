import pandas as pd
import numpy as np
import math
from helper import sample_activity, get_activity_matrix
from student_simulator import train_on_obs

PATH_TO_ACTIVITY_DIFFICULTY = "Data/Code Drop 2 Matrices.xlsx"
LITERACY_SHEET_NAME = 'Literacy (with levels as rows)'
MATH_SHEET_NAME = 'Math (with levels as rows)'
STORIES_SHEET_NAME = 'Stories'

# Train student simulator
activity_bkt, activity_to_kc_dict, skill_to_number_map, student_id_to_number_map = train_on_obs(1.0)

literacy_matrix, math_matrix, stories_matrix = get_activity_matrix(PATH_TO_ACTIVITY_DIFFICULTY, LITERACY_SHEET_NAME, MATH_SHEET_NAME, STORIES_SHEET_NAME)

# literacy_mu = 1.5
# width 3 for variance 0.112 width 1 for variance 0.014
# literacy_variance = 0.112
# run_literacy_placement(literacy_mu, literacy_variance, name="literacy")



# -----------------------------------

# gamma = 0.3
# literacy_mu = 1.5
# # width 3 for variance 0.112 width 1 for variance 0.014
# literacy_variance = 0.112
# literacy_sigma = literacy_variance**0.5

# literacy_row = 0
# literacy_col = 0
# alpha = 30
# beta = 1
# prev_mu = 0
# change_variance = False

# correct_threshold = 0.5
# min_threshold = 0.65

# student_id = "GPRXFZ_124"
# placement = None
    
# for i in range(15):
#     literacy_sigma = literacy_variance**0.5
#     new_row, new_col, activity = sample_activity(literacy_mu, literacy_sigma, literacy_matrix, literacy_row, lit_nan_idxs)
    
#     literacy_row = new_row
#     literacy_col = new_col
    
#     skills = []
#     skill = []
#     for kc in activity_to_kc_dict[activity]:
#         skill.append(skill_to_number_map[kc])

#     skills.append(skill)
#     student_ids = [student_id_to_number_map[student_id]] * len(skills)
    
#     corrects, min_corrects = activity_bkt.predict_percent_correct(student_ids, skills)
#     correct = corrects[0]
#     min_correct = min_corrects[0]
    
#     hist.append(correct - correct_threshold)
#     score = 0.0
#     for i in range(len(hist)):
#         score += (gamma**(len(hist) - 1 - i)) * hist[i] 
#     score = score/min(len(hist), 5)
#     prev_mu = literacy_mu
#     literacy_mu += alpha * score
#     deviation = math.floor(literacy_mu) - new_col
#     print(correct)
    
#     if correct < min_threshold:
#         placement = math.floor(prev_mu)
#         print("PLACED AT: ", math.floor(prev_mu))
#         break
    
#     if change_variance == True and deviation > 0:
#         if correct < correct_threshold:
#             literacy_variance += beta * 0.04
#     elif change_variance == True and deviation < 0:
#         if correct >= correct_threshold:
#             literacy_variance += beta * 0.04

#     print("MU: ", math.floor(literacy_mu*100)/100, "VARIANCE: ", math.floor(literacy_variance*100)/100,"DEVIATION: ", deviation, "%correct: ", correct)
        
#     if prev_mu > literacy_mu:
#         placement = math.floor(prev_mu)
#         print("PLACED AT: ", math.floor(prev_mu))
#         break

# if placement is None:
#     placement = prev_mu
    