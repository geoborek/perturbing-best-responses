import numpy as np
from scipy.optimize import linprog

def solve_game(M):
    A_ub = np.ones((M.shape[0],M.shape[1]+1))
    A_ub[:,1:] = -M
    b_ub = np.zeros(M.shape[0])
    c = np.zeros(M.shape[1]+1)
    c[0] = -1
    A_eq = np.ones((1,M.shape[1]+1))
    A_eq[0,0] = 0
    b_eq = np.array([[1]])
    bounds = [(None,None)] + [(0,1)]*(M.shape[1])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # print(A_ub, b_ub, c, A_eq, b_eq, bounds)            
    # print(res)
    # print(res.ineqlin.marginals)
    return res.fun, -res.ineqlin.marginals, res.x[1:]

if __name__ == "__main__":
    # Rock, Scissor, Paper
    # m = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])

    m = np.array([[30, -10, 20], [-10, 20, -20]])
    ep = 0.1
    m = np.array([[1, 1-ep, 1-ep], [0, 1, 1-ep], [0, 0, 1]])
    m = np.array([[1, 1, 1, 1, 0], 
                  [1, 0, 0, 0, 1], 
                  [0, 1, 0, 0, 1], 
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1]])

    # m = np.random.randint(0,2,(6,10)) 
    print(m)
    # m = np.array([[-1, 0, -1], [2, 1, 2], [-1, 0, -1]])
    # col player is maximizing
    # row player is minimizing
    print(solve_game(m))