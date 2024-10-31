'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    
    P = np.zeros((model.M, model.N, 4, model.M, model.N))

    for r in range(model.M):
        for c in range(model.N):
            nonzero_moves = [(r+1, c), (r-1, c), (r, c-1), (r, c+1)]
            for a in range(4):
                for tuple in nonzero_moves: 
                    r_p = tuple[0]  
                    c_p = tuple[1]
                    LEFT = (c_p == c - 1)
                    UP = (r_p == r - 1)
                    RIGHT = (c_p == c + 1)
                    DOWN = (r_p == r + 1)
                    # 0 left, 1 up, 2 right, 3 down
                    DIRECTION_MATCHES_ACTION = (a == 0 and LEFT) or (a == 1 and UP) or (a == 2 and RIGHT) or (a == 3 and DOWN)
                    DIRECTION_COUNTER_CLOCKWISE_ACTION = (a == 0 and DOWN) or (a == 1 and LEFT) or (a == 2 and UP) or (a == 3 and RIGHT)
                    DIRECTION_CLOCKWISE_ACTION = (a == 0 and UP) or (a == 1 and RIGHT) or (a == 2 and DOWN) or (a == 3 and LEFT)
                    OUT_OF_BOUNDS_OR_WALL = r_p < 0 or r_p == model.M or c_p < 0 or c_p == model.N or model.W[r_p, c_p]

                    if (model.TS[r, c]):
                        P[r, c, :, :, :] = 0 
                        continue


                    if DIRECTION_MATCHES_ACTION:
                        if OUT_OF_BOUNDS_OR_WALL:
                            P[r, c, a, r, c] += model.D[r, c, 0]
                        else:
                            P[r, c, a, r_p, c_p] = model.D[r, c, 0]
                    elif DIRECTION_COUNTER_CLOCKWISE_ACTION:
                        if OUT_OF_BOUNDS_OR_WALL:
                            P[r, c, a, r, c] += model.D[r, c, 1]
                        else:
                            P[r, c, a, r_p, c_p] = model.D[r, c, 1]
                    elif DIRECTION_CLOCKWISE_ACTION:
                        if OUT_OF_BOUNDS_OR_WALL:
                            P[r, c, a, r, c] += model.D[r, c, 2]
                        else:
                            P[r, c, a, r_p, c_p] = model.D[r, c, 2]
                    else:
                        if OUT_OF_BOUNDS_OR_WALL:
                            P[r, c, a, r, c] += 0
                        else:
                            P[r, c, a, r_p, c_p] = 0
    return P
    #raise RuntimeError("You need to write this part!")

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros((model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            U_next[r, c] += model.R[r, c]
            sum_prob = 0
            sum_prob_max = 0
            for a in range(4):
                for r_p in range(model.M):
                    for c_p in range(model.N):
                        sum_prob += P[r, c, a, r_p, c_p] * U_current[r_p, c_p]
                if sum_prob > sum_prob_max:
                    sum_prob_max = sum_prob
                sum_prob = 0
            U_next[r, c] += model.gamma * sum_prob_max
    
    return U_next

    #raise RuntimeError("You need to write this part!")

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition(model)
    U_start = np.zeros((model.M, model.N))
    U_current = compute_utility(model, U_start, P)

    converged = 1
    for r in range(model.M):
        for c in range(model.N):
            if abs(U_current[r, c] - U_start[r, c]) >= epsilon:
                converged = 0

    iter = 0
    U_next = 0
    while not converged and iter < 100:
        U_next = compute_utility(model, U_current, P)
        for r in range(model.M):
            for c in range(model.N):
                if abs(U_next[r, c] - U_current[r, c]) >= epsilon:
                    converged = 0
        U_current = U_next
        iter = iter + 1
    
    return U_next
    # raise RuntimeError("You need to write this part!")

def compute_policy_utility(model, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros((model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            U_next[r, c] += model.R[r, c]
            sum_prob = 0
            for r_p in range(model.M):
                for c_p in range(model.N):
                    sum_prob += model.FP[r, c, r_p, c_p] * U_current[r_p, c_p]
            U_next[r, c] += model.gamma * sum_prob
    
    return U_next

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    U_start = np.zeros((model.M, model.N))
    U_current = compute_policy_utility(model, U_start)

    converged = 1
    for r in range(model.M):
        for c in range(model.N):
            if abs(U_current[r, c] - U_start[r, c]) >= epsilon:
                converged = 0

    iter = 0
    U_next = 0
    while not converged and iter < 1000:
        U_next = compute_policy_utility(model, U_current)
        for r in range(model.M):
            for c in range(model.N):
                if abs(U_next[r, c] - U_current[r, c]) >= epsilon:
                    converged = 0
        U_current = U_next
        iter = iter + 1
    
    return U_next

    #raise RuntimeError("You need to write this part!")
