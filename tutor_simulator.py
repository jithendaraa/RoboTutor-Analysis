from reader import *
import random
import numpy as np

class TutorSimulator:
    def __init__(self, t1=None, t2=None, t3=None, area_rotation='L-N-L-S', area_rot_start=0, type=None, area_rotation_constraint=True, transition_constraint=True, thresholds=True):
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
        self.first_question = {'L': True,'N': True,'S': True}
        self.set_thresholds(t1, t2, t3) # Performance thresholds if this 'type' uses thresholds
        self.get_matrix_specific_activities()
        
        for i in range(area_rot_start):
            self.area_rotation = self.area_rotation[1:].append(self.area_rotation[0])

        if 'story.hear::Garden_Song.1' in self.story_activities:
            self.story_activities.remove('story.hear::Garden_Song.1')
        if 'story.hear::Safari_Song.1' in self.story_activities:
            self.story_activities.remove('story.hear::Safari_Song.1')

    
    def get_matrix_specific_activities(self):

        self.literacy_activities = []
        for i in range(len(self.literacy_matrix)):
            row = self.literacy_matrix[i]
            row_acts = row[:self.literacy_counts[i]]
            if len(self.literacy_activities) == 0:
                self.literacy_activities = row_acts
            else:
                self.literacy_activities = self.literacy_activities + row_acts
        
        self.math_activities = []
        for i in range(len(self.math_matrix)):
            row = self.math_matrix[i]
            row_acts = row[:self.math_counts[i]]
            if len(self.math_activities) == 0:
                self.math_activities = row_acts
            else:
                self.math_activities = self.math_activities + row_acts
        
        self.story_activities = []
        for i in range(len(self.stories_matrix)):
            row = self.stories_matrix[i]
            row_acts = row[:self.stories_counts[i]]
            if len(self.story_activities) == 0:
                self.story_activities = row_acts
            else:
                self.story_activities = self.story_activities + row_acts
        
        self.literacy_activities = pd.unique(np.array(self.literacy_activities)).tolist()
        self.math_activities = pd.unique(np.array(self.math_activities)).tolist()
        self.story_activities = pd.unique(np.array(self.story_activities)).tolist()

    
    def reset(self):
        self.attempt = 0
        self.first_question = {
            'L': True,
            'N': True,
            'S': True
        }

    def set_thresholds(self, t1, t2, t3):
        if self.type == 1 or self.type == 2:
            self.t1 = t1
            self.t2 = t2
            self.t3 = t3

    def prev(self, matrix_type, update_pos=True):
        # x,y are the current position indices in matrix 'matrix_type'
        if matrix_type == 'literacy':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]

            if x == 0 and y == 0:
                return 0, 0, self.literacy_matrix[0][0] 
            elif y > 0:
                if update_pos: self.literacy_pos[1] -= 1
                return x, y-1, self.literacy_matrix[x][y-1] 
            elif y == 0:
                if update_pos:
                    self.literacy_pos[0] -= 1
                    self.literacy_pos[1] = self.literacy_counts[x-1] - 1
                y = self.literacy_counts[x-1] - 1
                return x-1, y, self.literacy_matrix[x-1][y]

        elif matrix_type == 'math':
            x = self.math_pos[0]
            y = self.math_pos[1]
            if x == 0 and y == 0:
                return 0, 0, self.math_matrix[0][0] 
            elif y > 0:
                if update_pos:  self.math_pos[1] -= 1
                return x, y-1, self.math_matrix[x][y-1] 
            elif y == 0:
                if update_pos:
                    self.math_pos[0] -= 1
                    self.math_pos[1] = self.math_counts[x-1] - 1
                y = self.math_counts[x-1] - 1
                return x-1, y, self.math_matrix[x-1][y]
        
        elif matrix_type == 'stories':
            x = self.stories_pos[0]
            y = self.stories_pos[1]
            if x == 0 and y == 0:
                return 0, 0, self.stories_matrix[0][0] 
            elif y > 0:
                if update_pos:  self.stories_pos[1] -= 1
                return x, y-1, self.stories_matrix[x][y-1] 
            elif y == 0:
                if update_pos:
                    self.stories_pos[0] -= 1
                    self.stories_pos[1] = self.stories_counts[x-1] - 1
                y = self.stories_counts[x-1] - 1
                return x-1, y, self.stories_matrix[x-1][y]
    
    def same(self, matrix_type, update_pos=True):
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

    def next(self, matrix_type, update_pos=True):
        # x,y are the current position indices in matrix 'matrix_type'
        if matrix_type == 'literacy':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]
            
            if x == len(self.literacy_counts) - 1 and y == self.literacy_counts[-1] - 1:
                return x, y, self.literacy_matrix[x][y] 
            elif y < self.literacy_counts[x] - 1:
                if update_pos:  self.literacy_pos[1] += 1
                return x, y+1, self.literacy_matrix[x][y+1] 
            else:
                if update_pos:
                    self.literacy_pos[1] = 0
                    self.literacy_pos[0] += 1
                return x+1, 0, self.literacy_matrix[x+1][0]

        elif matrix_type == 'math':
            x = self.math_pos[0]
            y = self.math_pos[1]
            if x == len(self.math_counts) - 1 and y == self.math_counts[-1] - 1:
                return x, y, self.math_matrix[x][y] 
            elif y < self.math_counts[x] - 1:
                if update_pos:  self.math_pos[1] += 1
                return x, y+1, self.math_matrix[x][y+1] 
            else:
                if update_pos:
                    self.math_pos[1] = 0
                    self.math_pos[0] += 1
                return x+1, 0, self.math_matrix[x+1][0]
        
        elif matrix_type == 'stories':
            x = self.stories_pos[0]
            y = self.stories_pos[1]
            if x == len(self.stories_counts) - 1 and y == self.stories_counts[-1] - 1:
                return x, y, self.stories_matrix[x][y] 
            elif y < self.stories_counts[x] - 1:
                if update_pos:  self.stories_pos[1] += 1
                return x, y+1, self.stories_matrix[x][y+1] 
            else:
                if update_pos:
                    self.stories_pos[1] = 0
                    self.stories_pos[0] += 1
                return x+1, 0, self.stories_matrix[x+1][0]

    def next_next(self, matrix_type, update_pos=True):
        self.next(matrix_type, update_pos=update_pos)
        x, y, activity_name = self.next(matrix_type, update_pos=update_pos)   
        return x, y, activity_name 
    
    def get_matrix_area(self):
        self.attempt = self.attempt % len(self.area_rotation)
        area = self.area_rotation[self.attempt]
        if area == 'L': return 1
        elif area == 'N':   return 2
        elif area == 'S':   return 3
    
    def get_matrix_posn(self, p_know_act=None):
        self.attempt = self.attempt % len(self.area_rotation)
        prev_area = self.area_rotation[self.attempt - 1]
        area = self.area_rotation[self.attempt]
        
        if prev_area == area:
            if p_know_act >= self.t3:
                x, y, activity_name = self.next(area, False) if random.random() > 0.5 else self.next_next(area, False)
            elif p_know_act >= self.t2:
                x, y, activity_name = self.next(area, False)
            elif p_know_act >= self.t1:
                x, y, activity_name = self.same(area, False)
            else:
                x, y, activity_name = self.prev(area, False)
        
        else:
            if area == 'L':
                x, y = self.literacy_pos[0], self.literacy_pos[1]
            elif area == 'N':
                x, y = self.math_pos[0], self.math_pos[1]
            elif area == 'S':
                x, y = self.stories_pos[0], self.stories_pos[1]

        if area == 'L':
            matrix_posn = np.sum(self.literacy_counts[:x], dtype=np.uint8) + y + 1
        elif area == 'N':
            matrix_posn = np.sum(self.math_counts[:x], dtype=np.uint8) + y + 1
        elif area == 'S':
            matrix_posn = np.sum(self.stories_counts[:x], dtype=np.uint8) + y + 1

        return matrix_posn            
    
    def make_transition(self, prev_matrix_type, init_x, init_y, decision=None, response="?", p_know_prev_activity=None, prints=False):
        
        if self.thresholds:
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
        
        elif self.thresholds is not True:
            if decision == 'next_next':
                x, y, activity_name = self.next_next(prev_matrix_type)
            elif decision == 'next':
                x, y, activity_name = self.next(prev_matrix_type)
            elif decision == "same":
                x, y, activity_name = self.same(prev_matrix_type)
            elif decision == 'prev':
                x, y, activity_name = self.prev(prev_matrix_type)
            else:
                print("ERROR @tutor_simulator.py")
                print("Value for decision is:", decision)
        
        return x, y, activity_name

    def get_next_activity(self, decision=None, p_know_prev_activity=None, prev_activity_num=None, response="", prints=True):

        self.attempt = self.attempt % len(self.area_rotation)
        area = self.area_rotation[self.attempt] # area is the question that is going to be asked
        prev_area = self.area_rotation[self.attempt - 1]

        if self.type == 4:
            pass


        if area == 'L':
            matrix_type = 'literacy'
        elif area == 'N':
            matrix_type = 'math'
        elif area == 'S':
            matrix_type = 'stories'

        if prev_activity_num != None:
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
            
            x, y, activity_name = self.make_transition(prev_matrix_type, init_x, init_y, decision=decision, p_know_prev_activity=p_know_prev_activity, prints=prints)
            if self.first_question[area]:
                self.first_question[area] = False


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
        self.attempt += 1

        if activity_name[:23] == 'story.hear::Garden_Song':
            activity_name = activity_name[:23]
        
        elif activity_name[:23] == 'story.hear::Safari_Song':
            activity_name = activity_name[:23]

        return x, y, matrix_type, activity_name