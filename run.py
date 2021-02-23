import maze
import time

maze = maze.maze

testmaze = maze(50, 50)

testmaze.showCreation(4, 1000)

testmaze.showMazeSolver(1000, 4)

time.sleep(10)

quit()