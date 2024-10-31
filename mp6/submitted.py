# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

from queue import Queue
from queue import PriorityQueue
from collections import defaultdict

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function

    visited = set()
    path = []
    q = Queue()
    parents = {}
    q.put(maze.start)
    visited.add(maze.start)
    parents[maze.start] = None
    while not q.empty():
        node = q.get()
        if node == maze.waypoints[0]:
            break
        neighbors = maze.neighbors_all(node[0], node[1])
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = node
                q.put(neighbor)
    
    path = []
    end = maze.waypoints[0]
    while end is not None:
        path.append(end)
        end = parents[end]
    
    path.reverse()
    
    return path

def heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def reconstruct_path(maze, g):
    start = maze.start
    path = [maze.start]
    finished = set()
    finished.add(maze.start)
    while start != maze.waypoints[0]:
        neighbors = maze.neighbors_all(start[0], start[1])
        min = neighbors[0]
        for node in neighbors:
            if node not in finished:
                if (g[node] + heuristic(node, maze.waypoints[0])) < (g[min] + heuristic(min, maze.waypoints[0])):
                    min = node

        path.append(min)
        finished.add(min)
        start = min
    
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    pq = PriorityQueue()
    parents = {maze.start : None}
    
    gcost = defaultdict(lambda: float('inf'))
    gcost[maze.start] = 0
    
    fcost = defaultdict(lambda: float('inf'))
    fcost[maze.start] = heuristic(maze.start, maze.waypoints[0])
    
    pq.put((fcost[maze.start], maze.start))
    
    while not pq.empty():
        node = pq.get()[1]
        if node == maze.waypoints[0]:
            break
        neighbors = maze.neighbors_all(node[0], node[1])
        for neighbor in neighbors:
            cur_gcost = gcost[node] + 1
            if cur_gcost < gcost[neighbor]:
                parents[neighbor] = node
                gcost[neighbor] = cur_gcost
                fcost[neighbor] = cur_gcost + heuristic(neighbor, maze.waypoints[0])
                
                neighbor_in_pq = False
                
                pq_list = list(pq.queue)
                
                for item in pq_list:
                    if item[1] == neighbor:
                        neighbor_in_pq = True
                        
                if not neighbor_in_pq:
                    pq.put((fcost[neighbor], neighbor))
    
    path = []
    end = maze.waypoints[0]
    while end in parents:
        path.append(end)
        end = parents[end]
    
    path.reverse()
    
    return path

def astar_single_alt_visited(maze):
    pq = PriorityQueue()
    g = defaultdict(lambda: float('inf'))
    closed = set()
    g[maze.start] = 0
    parents = {maze.start: None}
    visited = set()
    visited.add(maze.start)

    # cost, point
    f_initial = g[maze.start]+heuristic(maze.start,maze.waypoints[0])
    pq.put((f_initial, maze.start))
    while not pq.empty():
        n = pq.get()[1]

        if n == maze.waypoints[0]:
            break

        closed.add(n)

        neighbors = maze.neighbors_all(n[0], n[1])
        for m in neighbors:
            if m not in closed:
                if g[n] + 1 < g[m]:
                    parents[m] = n
                g[m] = min(g[m], g[n] + 1)
                f = g[m]+heuristic(m, maze.waypoints[0])

                if m not in visited:
                    pq.put((f, m))
                    visited.add(m)

            if m in closed:
                if g[n] + 1 < g[m]:
                    closed.remove(m)


    path = []
    end = maze.waypoints[0]
    while end in parents:
        path.append(end)
        end = parents[end]

    path.reverse()

    return path

def astar_single_alt(maze, start, waypt):
    """
    Alternate implementation of the A* algorithm
    """
    #TODO: Implement astar_single

    pq = PriorityQueue()
    g = defaultdict(lambda: float('inf'))
    closed = set()
    g[start] = 0
    parents = {start: None}
    m_in_pq = False

    # cost, point
    f_initial = g[start]+heuristic(start,waypt)
    pq.put((f_initial, start))
    while not pq.empty():
        n = pq.get()[1]

        if n == waypt:
            break

        closed.add(n)

        neighbors = maze.neighbors_all(n[0], n[1])
        for m in neighbors:
            if m not in closed:
                if g[n] + 1 < g[m]:
                    parents[m] = n
                g[m] = min(g[m], g[n] + 1)
                f = g[m]+heuristic(m, waypt)
                
                pq_list = list(pq.queue)
                for item in pq_list:
                    if item[1] == m:
                        m_in_pq = True 
                if not m_in_pq:
                    pq.put((f, m))
                m_in_pq = False

            if m in closed:
                if g[n] + 1 < g[m]:
                    closed.remove(m)
        
    
    path = []
    end = waypt
    while end in parents:
        path.append(end)
        end = parents[end]
    
    path.reverse()
    
    return path


def prim_mst_length(waypts):

    # initialize waypoints graph
    graph = {waypoint : [] for waypoint in waypts}
    for waypoint_1 in waypts:
        for waypoint_2 in waypts:
            if waypoint_1 == waypoint_2:
                continue
            graph[waypoint_1].append((waypoint_2, heuristic(waypoint_1, waypoint_2)))
    
    start_vertex = list(graph.keys())[0]
    visited = set([start_vertex])
    edges = [(cost, start_vertex, to) for to, cost in graph[start_vertex]]

    mst = []

    while edges:
        cost, frm, to = min(edges)
        edges.remove((cost, frm, to))

        if to not in visited:
            visited.add(to)
            mst.append((frm, to, cost))

            for to_next, cost in graph[to]:
                if to_next not in visited:
                    edges.append((cost, to, to_next))
    
    length = 0

    visited = set()
    
    for node in mst:
        if frozenset((node[0], node[1])) not in visited:
            visited.add(frozenset((node[0], node[1])))
            length += node[2]

    return length



def distance(x, y):
    return ((x[0] - y[0])^2 + (x[1] - y[1])^2)^0.5

def closest_waypoint(waypts, coord):
    min = heuristic(coord, waypts[0])
    for waypoint in waypts:
        if heuristic(coord, waypoint) < min:
            min = heuristic(coord, waypoint)
    return min

    

def heuristic_ec(coord, waypts):
    return closest_waypoint(waypts, coord) + prim_mst_length(waypts)

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    pq = PriorityQueue()
    g = defaultdict(lambda: float('inf'))
    closed = set()
    g[maze.start] = 0
    parents = {maze.start: None}
    visited = set()
    visited.add(maze.start)

    paths = []


    # 0 means unvisited, 1 means visited
    waypoint_status = [0 for i in range(len(maze.waypoints))]

    visited_status = {maze.start : waypoint_status}

    # get index of waypoint in the list
    waypoint_indexing = {waypoint : j for j, waypoint in enumerate(maze.waypoints)}
    waypoint_reverse_indexing = {j : waypoint for j, waypoint in enumerate(maze.waypoints)}

    # set of waypoints
    waypoint_set = set(maze.waypoints)

    last_waypoint = maze.waypoints[0]
    j = 0

    # def heuristic_ec(coord, waypts):
    #     return closest_waypoint(waypts, coord) + prim_mst_length(waypts)

    # f_cost, point, waypoint_status
    f_initial = g[maze.start]+heuristic_ec(maze.start,maze.waypoints)
    pq.put((f_initial, maze.start, waypoint_status))
    while not pq.empty():
        n = pq.get()[1]

        if n in waypoint_indexing and waypoint_status[waypoint_indexing[n]] == 0: 
            waypoint_status[waypoint_indexing[n]] = 1 # waypoint has been visited
            waypoint_set.remove(n) # remove from set
            last_waypoint = n
            path = []
            end = n
            while end in parents:
                path.append(end)
                end = parents[end]
            path.reverse()
            if j != 0:
                path = path[1:]
            j += 1
            paths.append(path)


        if len(waypoint_set) == 0:
            break

        closed.add(n)

        neighbors = maze.neighbors_all(n[0], n[1])
        for m in neighbors:
            if m not in closed:
                if g[n] + 1 < g[m]:
                    parents[m] = n
                g[m] = min(g[m], g[n] + 1)
                f = g[m]+heuristic_ec(m, list(waypoint_set))

                if m not in visited:
                    pq.put((f, m, waypoint_status))
                    visited.add(m)
                    visited_status[m] = waypoint_status
                
                if m in visited:
                    if visited_status[m] != waypoint_status:
                        pq.put((f, m, waypoint_status))

            if m in closed:
                if g[n] + 1 < g[m]:
                    closed.remove(m)


    # path = []
    # end = last_waypoint
    # while end in parents:
    #     path.append(end)
    #     end = parents[end]

    # path.reverse()
    answer = []
    for path in paths:
        for coord in path:
            answer.append(coord)

    return answer


def astar_multiple_greedy(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    waypoints_set = set(maze.waypoints)
    current_pos = maze.start
    paths = []

    i = 0


    while len(waypoints_set) != 0:
        waypoints_list = list(waypoints_set)
        min_waypt = waypoints_list[0]
        for waypt in waypoints_list:
            if heuristic(current_pos, waypt) < heuristic(current_pos, min_waypt):
                min_waypt = waypt
        waypoints_set.remove(min_waypt)
        cur_path = astar_single_alt(maze, current_pos, min_waypt)
        if i != 0:
            cur_path = cur_path[1:]
        i += 1
        paths.append(cur_path)
        current_pos = min_waypt
    
    answer = []
    
    for path in paths:
        for point in path:
            answer.append(point)
    
    return answer

    
def astar_multiple_broken(maze):
    pq = PriorityQueue()
    parents = {maze.start : None}
    
    gcost = defaultdict(lambda: float('inf'))
    gcost[maze.start] = 0
    
    fcost = defaultdict(lambda: float('inf'))
    fcost[maze.start] = heuristic_ec(maze.start, maze.waypoints)
    
    pq.put((fcost[maze.start], maze.start))

    waypoints_set = set()
    for waypt in maze.waypoints:
        waypoints_set.add(waypt)
    
    while not pq.empty():
        node = pq.get()[1]


        if node in waypoints_set:
            waypoints_set.remove(node)

            if len(waypoints_set) == 0:
                break

    
        neighbors = maze.neighbors_all(node[0], node[1])
        for neighbor in neighbors:
            cur_gcost = gcost[node] + 1
            if cur_gcost < gcost[neighbor]:
                parents[neighbor] = node
                gcost[neighbor] = cur_gcost
                fcost[neighbor] = cur_gcost + heuristic_ec(neighbor, list(waypoints_set))
                
                neighbor_in_pq = False
                
                pq_list = list(pq.queue)
                
                for item in pq_list:
                    if item[1] == neighbor:
                        neighbor_in_pq = True
                        
                if not neighbor_in_pq:
                    pq.put((fcost[neighbor], neighbor))
    
    path = []
    end = maze.waypoints[0]
    while end in parents:
        path.append(end)
        end = parents[end]
    
    path.reverse()
    
    return path
