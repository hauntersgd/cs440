import maze
import submitted
import importlib

maze3 =  maze.Maze('data/part-3/tiny')

importlib.reload(submitted)

#graph, mst, length = submitted.prim_mst_length(maze3.waypoints)

# gdict = submitted.astar_single(maze3)

with open('output.txt', 'w') as file:
    # for key,value in graph.items():
    #     print(f"{key} : {value}", file=file)
    print(maze3.waypoints, file = file)
    print(maze3.start, file = file)
    print(maze3.neighbors_all(maze3.start[0], maze3.start[1]), file = file)

    # for node in mst:
    #         print(node, file=file)

    # print(maze3.start, file = file)

    # print("")

    # values = list(gdict.values())
    # points = list(gdict.keys())
    # for point, value in zip(points, values):
    #     print(f"{point} : {value + submitted.heuristic(point, maze3.waypoints[0])}", file = file)



    #print(length, file=file)