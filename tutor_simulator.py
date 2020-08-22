from reader import *
import random

class TutorSimulator:
    def __init__(self, t1, t2, t3, area_rotation='L-N-L-S', area_rot_start=0, type=None, area_rotation_constraint=True, transition_constraint=True, thresholds=True):
        self.literacy_matrix, self.math_matrix, self.stories_matrix, self.literacy_counts, self.math_counts, self.stories_counts = read_activity_matrix()
        self.literacy_pos = [0,0]
        self.math_pos = [0,0]
        self.stories_pos = [0,0]
        self.type = type
        self.thresholds = thresholds

        self.area_rotation_constraint = area_rotation_constraint
        self.transition_constraint = transition_constraint
        self.area_rotation = area_rotation.split('-')
        self.attempt = 0

        for i in range(area_rot_start):
            self.area_rotation = self.area_rotation[1:].append(self.area_rotation[0])
        
        self.first_question = {
            'L': True,
            'N': True,
            'S': True
        }
        
        # Performance thresholds if this TutorSimulator type uses thresholds
        self.set_thresholds(t1, t2, t3)
    
    def reset(self):
        self.attempt = 0
        self.first_question = {
            'L': True,
            'N': True,
            'S': True
        }

    def set_thresholds(self, t1, t2, t3):
        
        if self.type == 1:
            self.t1 = t1
            self.t2 = t2
            self.t3 = t3

    def prev(self, matrix_type):
        # x,y are the current position indices in matrix 'matrix_type'
        if matrix_type == 'literacy':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]

            if x == 0 and y == 0:
                return 0, 0, self.literacy_matrix[0][0] 
            elif y > 0:
                self.literacy_pos[1] -= 1
                return x, y-1, self.literacy_matrix[x][y-1] 
            elif y == 0:
                self.literacy_pos[0] -= 1
                self.literacy_pos[1] = self.literacy_counts[x-1] - 1
                y = self.literacy_pos[1]
                return x-1, y, self.literacy_matrix[x-1][y]

        elif matrix_type == 'math':
            x = self.math_pos[0]
            y = self.math_pos[1]
            if x == 0 and y == 0:
                return 0, 0, self.math_matrix[0][0] 
            elif y > 0:
                self.math_pos[1] -= 1
                return x, y-1, self.math_matrix[x][y-1] 
            elif y == 0:
                self.math_pos[0] -= 1
                self.math_pos[1] = self.math_counts[x-1] - 1
                y = self.math_pos[1]
                return x-1, y, self.math_matrix[x-1][y]
        
        elif matrix_type == 'stories':
            x = self.stories_pos[0]
            y = self.stories_pos[1]
            if x == 0 and y == 0:
                return 0, 0, self.stories_matrix[0][0] 
            elif y > 0:
                self.stories_pos[1] -= 1
                return x, y-1, self.stories_matrix[x][y-1] 
            elif y == 0:
                self.stories_pos[0] -= 1
                self.stories_pos[1] = self.stories_counts[x-1] - 1
                y = self.stories_pos[1]
                return x-1, y, self.stories_matrix[x-1][y]
    
    def same(self, matrix_type):
        if matrix_type == 'literacy':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]
            activity_num = self.literacy_matrix[x][y]

        elif matrix_type == 'math':
            x = self.math_pos[0]
            y = self.math_pos[1]
            activity_num = self.math_matrix[x][y]

        elif matrix_type == 'stories':
            x = self.stories_pos[0]
            y = self.stories_pos[1]
            activity_num = self.stories_matrix[x][y]
        
        return x, y, activity_num

    def next(self, matrix_type):
        # x,y are the current position indices in matrix 'matrix_type'
        if matrix_type == 'literacy':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]
            
            if x == len(self.literacy_counts) - 1 and y == self.literacy_counts[-1] - 1:
                return x, y, self.literacy_matrix[x][y] 
            elif y < self.literacy_counts[x] - 1:
                self.literacy_pos[1] += 1
                return x, y+1, self.literacy_matrix[x][y+1] 
            else:
                self.literacy_pos[1] = 0
                self.literacy_pos[0] += 1
                return x+1, 0, self.literacy_matrix[x+1][0]

        elif matrix_type == 'math':
            x = self.math_pos[0]
            y = self.math_pos[1]
            if x == len(self.math_counts) - 1 and y == self.math_counts[-1] - 1:
                return x, y, self.math_matrix[x][y] 
            elif y < self.math_counts[x] - 1:
                self.math_pos[1] += 1
                return x, y+1, self.math_matrix[x][y+1] 
            else:
                self.math_pos[1] = 0
                self.math_pos[0] += 1
                return x+1, 0, self.math_matrix[x+1][0]
        
        elif matrix_type == 'stories':
            x = self.stories_pos[0]
            y = self.stories_pos[1]
            if x == len(self.stories_counts) - 1 and y == self.stories_counts[-1] - 1:
                return x, y, self.stories_matrix[x][y] 
            elif y < self.stories_counts[x] - 1:
                self.stories_pos[1] += 1
                return x, y+1, self.stories_matrix[x][y+1] 
            else:
                self.stories_pos[1] = 0
                self.stories_pos[0] += 1
                return x+1, 0, self.stories_matrix[x+1][0]

    def next_next(self, matrix_type):
        self.next(matrix_type)
        x, y, activity_name = self.next(matrix_type)   
        return x, y, activity_name 
    
    def get_next_activity(self, p_know_prev_activity=None, prev_activity_num=None, response="", prints=True):

        self.attempt = self.attempt % len(self.area_rotation)
        area = self.area_rotation[self.attempt]
        prev_area = self.area_rotation[self.attempt - 1]

        if area == 'L':
            matrix_type = 'literacy'
        elif area == 'N':
            matrix_type = 'math'
        elif area == 'S':
            matrix_type = 'stories'

        if prev_activity_num != None and self.thresholds:
            prev_matrix_type = 'literacy'
            init_x = self.literacy_pos[0]
            init_y = self.literacy_pos[1]
            if prev_area == 'N':
                prev_matrix_type = 'math'
                init_x = self.math_pos[0]
                init_y = self.math_pos[1]
            if prev_area == 'S':
                prev_matrix_type = 'stories'
                init_x = self.stories_pos[0]
                init_y = self.stories_pos[1]

            if p_know_prev_activity >= self.t3:
                if random.random() > 0.5:
                    if prints: print("NEXT")
                    x, y, activity_name = self.next(prev_matrix_type)
                else:
                    if prints: print("NEXT_NEXT")
                    x, y, activity_name = self.next_next(prev_matrix_type)
            elif p_know_prev_activity >= self.t2:
                if prints: print("NEXT")
                x, y, activity_name = self.next(prev_matrix_type)
            elif p_know_prev_activity >= self.t1:
                x, y, activity_name = self.same(prev_matrix_type)
            else:
                x, y, activity_name = self.prev(prev_matrix_type)
            if prints: print('PREV MATRIX POSN (' + str(prev_matrix_type) + "): move from [" + str(init_x) + "," + str(init_y) + "] -> [" + str(x) + "," + str(y) + "], P(Know prev activity): " + str(p_know_prev_activity) + " Response: " + response)

        if area == 'L':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]
            activity_name = self.literacy_matrix[x][y]
        elif area == 'N':
            x = self.math_pos[0]
            y = self.math_pos[1]
            activity_name = self.math_matrix[x][y]
        elif area == 'S':
            x = self.stories_pos[0]
            y = self.stories_pos[1]
            activity_name = self.stories_matrix[x][y]

        if self.first_question[area]:
            self.first_question[area] = False
        
        self.attempt += 1

        if activity_name[:23] == 'story.hear::Garden_Song':
            activity_name = activity_name[:23]

        return x, y, matrix_type, activity_name