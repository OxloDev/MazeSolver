import numpy as np
import cv2
import random


class Cell:
    def __init__(self, x, y, parent_cel):
        self.x = x
        self.y = y
        self.parent_cell = parent_cel
        self.children_cells = []
        self.visited = False
    #Backtracking Array
    def sendImgArr(self):
        img_arr = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])

        for child in self.children_cells:
            x, y = child.x, child.y
            img_arr[1 + (y - self.y), 1 + (x - self.x)] = 1

        if self.parent_cell is not None:
            x, y = self.parent_cell.x, self.parent_cell.y
            img_arr[1 + (y - self.y), 1 + (x - self.x)] = 1

        return img_arr

class maze:
    #initializes maze
    def __init__(self, rows, columns):
        self.columns = columns
        self.rows = rows
        self.maze = np.empty((self.columns, self.rows), dtype=object)
        self.start = None
        self.end = None

    #find Neighboring cell
    def findNeighbors(self, cell):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []
        for direction in directions:
            new_x = cell.x + direction[0]
            new_y = cell.y + direction[1]

            if 0 <= new_x and new_x < self.columns:
                if 0 <= new_y and new_y < self.rows:
                    if self.maze[new_y, new_x] is None:
                        neighbors.append(Cell(new_x, new_y, None))
        return neighbors

    #Creates maze
    def mazeCreation(self):
        front = Cell(0, 0, None)
        #Starts from 0,0
        self.maze[0, 0] = front

        current_cell = front
        while current_cell is not None:
            neighbors = self.findNeighbors(current_cell)
            if len(neighbors) > 0:
                new_cell = random.choice(neighbors)
                self.maze[new_cell.y, new_cell.x] = new_cell
                new_cell.parent_cell = current_cell
                current_cell.children_cells.append(new_cell)
                current_cell = new_cell
            else:
                current_cell = current_cell.parent_cell
    

        self.start = front
        self.end = self.maze[self.rows - 1, self.columns - 1]

    #Shows creation to user
    def showCreation(self, scale, fps):
        maze_img_arr = np.zeros((self.maze.shape[0] * 3, self.maze.shape[1] * 3, 3))

        front = Cell(0, 0, None)
        self.maze[0, 0] = front

        current_cell = front
        while current_cell is not None:

            # finds neighboring cells
            neighbors = self.findNeighbors(current_cell)

            if len(neighbors) > 0:
                #creates a new cell from random neighbor
                new_cell = random.choice(neighbors)
                #adds new cell to the maze
                self.maze[new_cell.y, new_cell.x] = new_cell
                new_cell.parent_cell = current_cell
                #adds new cell to children cells
                current_cell.children_cells.append(new_cell)
                #draws current cell
                current_cell_img = (current_cell.sendImgArr().repeat(3).reshape((3, 3, 3))) * 255
                maze_img_arr[current_cell.y * 3: (current_cell.y * 3) + 3, current_cell.x * 3: (current_cell.x * 3) + 3] = current_cell_img
                #makes the new cell the "current cell"
                current_cell = new_cell
                #redraws current cell
                current_cell_img = (current_cell.sendImgArr().repeat(3).reshape((3, 3, 3))) * 255
                maze_img_arr[current_cell.y * 3: (current_cell.y * 3) + 3, current_cell.x * 3: (current_cell.x * 3) + 3] = current_cell_img
            else:
                #draws current cell as seen
                current_cell_img = (current_cell.sendImgArr().repeat(3).reshape((3, 3, 3))) * 170
                maze_img_arr[current_cell.y * 3: (current_cell.y * 3) + 3, current_cell.x * 3: (current_cell.x * 3) + 3] = current_cell_img
                #Backpropagates
                current_cell = current_cell.parent_cell

            maze_img_arr = maze_img_arr.astype(np.uint8)

            cv2.imshow("Maze", cv2.resize(maze_img_arr, (int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)), interpolation=0))
            if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                break

        self.start = front
        self.end = self.maze[self.rows - 1, self.columns - 1]
        maze_img_arr = (maze_img_arr == 170) * 255
        maze_img_arr[(self.start.y * 3) + 1: (self.start.y * 3) + 2, (self.start.x * 3) + 1: (self.start.x * 3) + 2, :] = np.array([0, 255, 0])
        maze_img_arr[(self.end.y * 3) + 1: (self.end.y * 3) + 2, (self.end.x * 3) + 1: (self.end.x * 3) + 2, :] = np.array([0, 0, 255])
        maze_img_arr = maze_img_arr.astype(np.uint8)
        cv2.imshow("Maze", cv2.resize(maze_img_arr, (int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)), interpolation=0))
        cv2.waitKey(0)
    
    #Shows maze to user
    def showMaze(self, scale):
        maze_img_arr = np.zeros((self.maze.shape[0] * 3, self.maze.shape[1] * 3))
        for row in range(self.maze.shape[0]):
            for column in range(self.maze.shape[1]):
                x, y = (column * 3, row * 3)
                maze_img_arr[y: y + 3, x: x + 3] = self.maze[row, column].sendImgArr()

        maze_img_arr = maze_img_arr * 255
        maze_img_arr = maze_img_arr.repeat(3).reshape((maze_img_arr.shape[0], maze_img_arr.shape[1], 3))
        maze_img_arr[(self.start.y * 3) + 1: (self.start.y * 3) + 2, (self.start.x * 3) + 1: (self.start.x * 3) + 2, :] = np.array([0, 255, 0])
        maze_img_arr[(self.end.y * 3) + 1: (self.end.y * 3) + 2, (self.end.x * 3) + 1: (self.end.x * 3) + 2, :] = np.array([0, 0, 255])
        maze_img_arr = cv2.resize(maze_img_arr, dsize=(int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)), interpolation=0)
        maze_img_arr = maze_img_arr.astype(np.uint8)

        cv2.imshow("Maze", maze_img_arr)
        cv2.waitKey(0)

    def sendImgArr(self):
        maze_img_arr = np.zeros((self.maze.shape[0] * 3, self.maze.shape[1] * 3))
        for row in range(self.maze.shape[0]):
            for column in range(self.maze.shape[1]):
                x, y = (column * 3, row * 3)
                maze_img_arr[y: y + 3, x: x + 3] = self.maze[row, column].sendImgArr()

        maze_img_arr = maze_img_arr * 255
        maze_img_arr = maze_img_arr.repeat(3).reshape((maze_img_arr.shape[0], maze_img_arr.shape[1], 3))
        maze_img_arr[(self.start.y * 3) + 1: (self.start.y * 3) + 2, (self.start.x * 3) + 1: (self.start.x * 3) + 2, :] = np.array([0, 255, 0])
        maze_img_arr[(self.end.y * 3) + 1: (self.end.y * 3) + 2, (self.end.x * 3) + 1: (self.end.x * 3) + 2, :] = np.array([0, 0, 255])
        maze_img_arr = maze_img_arr.astype(np.uint8)

        return maze_img_arr

    def possiblePaths(self, cell):
        paths = []
        for child in cell.children_cells:
            if not child.visited:
                paths.append(child)
        return paths

    def mazeSolver(self):
        front = self.start
        current_cell = front
        answer = [front]
        while current_cell != self.end:
            possible_paths = self.findPossiblePaths(current_cell)
            if len(possible_paths) > 0:
                path = random.choice(possible_paths)
                current_cell = path

                answer.append(current_cell)
            else:
                current_cell.visited = True
                answer.remove(current_cell)
                current_cell = current_cell.parent_cell

        return answer

    def showMazeSolver(self, fps, scale):
        maze_img_arr = self.sendImgArr()
        front = self.start
        current_cell = front
        answer = [front]
        while current_cell != self.end:
            possible_paths = self.possiblePaths(current_cell)
            #outlines pottential path
            if len(possible_paths) > 0:
                maze_img_arr[(current_cell.y * 3) + 1, (current_cell.x * 3) + 1, :] = np.array([158 ,255, 0])
                if current_cell.parent_cell is not None:
                    maze_img_arr[(current_cell.y * 3) + 1 + (current_cell.parent_cell.y - current_cell.y), (current_cell.x * 3) + 1 + (current_cell.parent_cell.x - current_cell.x), :] = np.array([158 ,255, 0])
                path = random.choice(possible_paths)
                maze_img_arr[(current_cell.y * 3) + 1 + (path.y - current_cell.y), (current_cell.x * 3) + 1 + (path.x - current_cell.x), :] = np.array([158 ,255, 0])
                current_cell = path
                maze_img_arr[(current_cell.y * 3) + 1, (current_cell.x * 3) + 1: (current_cell.x * 3) + 2, :] = np.array([158 ,255, 0])
                if current_cell.parent_cell is not None:
                    maze_img_arr[(current_cell.y * 3) + 1 + (current_cell.parent_cell.y - current_cell.y), (current_cell.x * 3) + 1 + (current_cell.parent_cell.x - current_cell.x), :] = np.array([158 ,255, 0])

                answer.append(current_cell)
            #flags bad path
            else:
                maze_img_arr[(current_cell.y * 3) + 1: (current_cell.y * 3) + 2, (current_cell.x * 3) + 1: (current_cell.x * 3) + 2, :] = np.array([175, 179, 255])
                if current_cell.parent_cell is not None:
                    maze_img_arr[(current_cell.y * 3) + 1 + (current_cell.parent_cell.y - current_cell.y), (current_cell.x * 3) + 1 + (current_cell.parent_cell.x - current_cell.x), :] = np.array([175, 179, 255])
                for child in current_cell.children_cells:
                    maze_img_arr[(current_cell.y * 3) + 1 + (child.y - current_cell.y), (current_cell.x * 3) + 1 + (child.x - current_cell.x), :] = np.array([175, 179, 255])

                current_cell.visited = True
                answer.remove(current_cell)
                current_cell = current_cell.parent_cell

                maze_img_arr[(current_cell.y * 3) + 1: (current_cell.y * 3) + 2, (current_cell.x * 3) + 1: (current_cell.x * 3) + 2, :] = np.array([175, 179, 255])
                for child in current_cell.children_cells:
                    maze_img_arr[(current_cell.y * 3) + 1 + (child.y - current_cell.y), (current_cell.x * 3) + 1 + (child.x - current_cell.x), :] = np.array([175, 179, 255])

            maze_img_arr = maze_img_arr.astype(np.uint8)

            cv2.imshow("Maze", cv2.resize(maze_img_arr,(int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)),interpolation=0))
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break

        maze_img_arr = self.sendImgArr()

        maze_img_arr[(self.end.y * 3) + 1: (self.end.y * 3) + 2, (self.end.x * 3) + 1: (self.end.x * 3) + 2, :] = np.array([175, 179, 255])

        answer.reverse()

        for i in range(len(answer)):
            cell = answer[i]
            maze_img_arr[(cell.y * 3) + 1, (cell.x * 3) + 1, :] = np.array([0, 255, 0])

            if (i + 1) < len(answer):
                next_cell = answer[i + 1]
            else:
                next_cell = cell

            maze_img_arr[(cell.y * 3) + 1 + (next_cell.y - cell.y), (cell.x * 3) + 1 + (next_cell.x - cell.x), :] = np.array([0, 255, 0])

            if (i - 1) >= 0:
                last_cell = answer[i - 1]
            else:
                last_cell = cell

            maze_img_arr[(cell.y * 3) + 1 + (last_cell.y - cell.y), (cell.x * 3) + 1 + (last_cell.x - cell.x), :] = np.array([0, 255, 0])


            cv2.imshow("Maze", cv2.resize(maze_img_arr, (int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)), interpolation=0))
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break


        cv2.imshow("Maze", cv2.resize(maze_img_arr, (int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)), interpolation=0))
        cv2.waitKey(1000)

        return answer

    #Shows maze solution
    def showMazeSolution(self, scale):
        maze_img_arr = self.sendImgArr()
        current_cell = self.end
        previous_cell = None

        while True:
            maze_img_arr[(current_cell.y * 3) + 1, (current_cell.x * 3) + 1, :] = np.array([0, 255, 0])
            if current_cell.parent_cell is not None:
                maze_img_arr[(current_cell.y * 3) + 1 + (current_cell.parent_cell.y - current_cell.y), (current_cell.x * 3) + 1 + current_cell.parent_cell.x - current_cell.x, :] = np.array([0, 255, 0])
            if previous_cell is not None:
                maze_img_arr[(current_cell.y * 3) + 1 + (previous_cell.y - current_cell.y), (current_cell.x * 3) + 1 + previous_cell.x - current_cell.x, :] = np.array([0, 255, 0])
            previous_cell = current_cell
            current_cell = current_cell.parent_cell

            if current_cell is None:
                break


        cv2.imshow("Maze", cv2.resize(maze_img_arr, (int(maze_img_arr.shape[1] * scale), int(maze_img_arr.shape[0] * scale)), interpolation=0))
        cv2.waitKey(0)