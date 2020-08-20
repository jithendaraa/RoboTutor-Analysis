from reader import *

class TutorSimulator:
    def __init__(self, t1, t2, t3, type=None):
        self.literacy_matrix, self.math_matrix, self.stories_matrix, self.literacy_counts, self.math_counts, self.stories_counts = read_activity_matrix()
        self.literacy_pos = [0,0]
        self.math_pos = [0,0]
        self.stories_pos = [0,0]
        self.type = type
        
        # Performance thresholds if this TutorSimulator type uses thresholds
        self.set_thresholds(t1, t2, t3)
    
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
                print("Already at first problem in", matrix_type, "matrix")
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
                print("Already at first problem in", matrix_type, "matrix")
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
                print("Already at first problem in", matrix_type, "matrix")
                return 0, 0, self.stories_matrix[0][0] 
            elif y > 0:
                self.stories_pos[1] -= 1
                return x, y-1, self.stories_matrix[x][y-1] 
            elif y == 0:
                self.stories_pos[0] -= 1
                self.stories_pos[1] = self.stories_counts[x-1] - 1
                y = self.stories_pos[1]
                return x-1, y, self.stories_matrix[x-1][y]

    
    def next(self, matrix_type):
        # x,y are the current position indices in matrix 'matrix_type'
        if matrix_type == 'literacy':
            x = self.literacy_pos[0]
            y = self.literacy_pos[1]
            
            if x == len(self.literacy_counts) - 1 and y == self.literacy_counts[-1] - 1:
                print('Already at the last problem in', matrix_type, 'matrix')
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
                print('Already at the last problem in', matrix_type, 'matrix')
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
                print('Already at the last problem in', matrix_type, 'matrix')
                return x, y, self.stories_matrix[x][y] 
            elif y < self.stories_counts[x] - 1:
                self.stories_pos[1] += 1
                return x, y+1, self.stories_matrix[x][y+1] 
            else:
                self.stories_pos[1] = 0
                self.stories_pos[0] += 1
                return x+1, 0, self.stories_matrix[x+1][0]

    def next_next(self, matrix_type):
        next(self, matrix_type)
        next(self, matrix_type)    
    



if __name__ == '__main__':
    tutor_simulator = TutorSimulator()