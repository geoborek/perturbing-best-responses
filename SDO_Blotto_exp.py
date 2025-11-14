import numpy as np
from itertools import combinations_with_replacement
from math import fsum
# import nashpy
import perturb_FP as solver
from zero_sum import solve_game 
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

def eval(x, y): # check which number is bigger (winning)
    if x>y:
        return 1
    elif x==y:
        return 0
    else:
        return -1

def payoff(str1, str2): # evaluate payoff for two strategies
    assert len(str1) == len(str2)
    elementwise_eval = map(eval, str1, str2) # check who wins elementwise
    return eval(fsum(elementwise_eval), 0) # check who wins in majority times

def blotto_strategies(bfs, resources):
    combs = list(combinations_with_replacement(range(1, resources+1), bfs))
    return list(filter(lambda x: (x==tuple(sorted(x))) & (fsum(x)==resources) , combs)) # non descending order of units assigned and units summing to resources alloted

def blotto_game(bfs, resources):
    strategies = blotto_strategies(bfs,resources)
    n = len(strategies)
    
    M = np.full((n, n),np.nan)
    
    for i in range(n):
       for j in range(n):
           M[i][j] = payoff(strategies[i],strategies[j])
    return M.T

EP = 0.1
TYPE = "uniform"
PARAMS = [0.01, 0.001]

# bfs = range(3,4)
# bfs = range(4,5)
bfs = range(5,6)
units = range(5, 50) 
REPETITIONS = 10

np.random.seed(1)

if __name__ == "__main__":
    for PARAM in PARAMS:

        means = np.zeros(len(bfs) * len(units))
        stds = np.zeros(len(bfs) * len(units))
        means_nop = np.zeros(len(bfs) * len(units))
        stds_nop = np.zeros(len(bfs) * len(units))

        for i, bf in tqdm(enumerate(bfs)):
            for j, unit in tqdm(enumerate(units)):
                index = i * len(units) + j
                iters = np.zeros(REPETITIONS)
                iters_nop = np.zeros(REPETITIONS)
                M = blotto_game(bf, unit)
                for r in range(REPETITIONS):
                    rval, cval, t, cne, rne = solver.DO_Nash(M, EP, PARAM, TYPE)
                    # print(rval, cval, t)
                    iters[r] = t
                rval, cval, t, cne, rne = solver.DO_Nash(M, EP, 0, "none")

                means[index] = iters.mean()
                stds[index] = iters.std()
                means_nop[index] = t
        
        base_str = f"{PARAM:.0e}"  # gives '1e-03'
        coeff, exp = base_str.split('e')
        exp = int(exp)
        new_label = fr"U(-{PARAM},{PARAM})"

        plt.plot(range(len(bfs) * len(units)), means, label = new_label)
        plt.fill_between(range(len(bfs) * len(units)), np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    plt.plot(range(len(bfs) * len(units)), means_nop, label = "No perturbation")
    
    plt.title(f"{bfs[0]} battlefields")
    plt.xlabel("units budget")
    plt.ylabel("iterations")
    plt.legend()
    plt.show()
    # plt.savefig(rf"C:\Users\tupol\Desktop\PhD\game theory\main\perturbing-best-responses\plots\SDO Blotto {bfs[0]} bfs, type {TYPE}.png")
    
    data = np.array([means, stds, means_nop])
    df = pd.DataFrame(data, columns=range(len(bfs) * len(units)), index=['means', 'stds', 'means_nop'])
    df.to_csv("experiments/SDO_Blotto.csv", mode='a')
