import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import game

EP = 0.01
TYPE = "normal"
PARAM = 0.0001
COEF = 10

SIZES = range(2, 15) 
REPETITIONS = 10

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))
    stds_nop = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):
        
        iters = np.zeros(REPETITIONS)
        iters_nop = np.zeros(REPETITIONS)
        for r in range(REPETITIONS):
            g = game.Grid(n, 5, COEF)
            rval, cval, t, cne, rne = game.DO_Nash(g, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t
        g = game.Grid(n, 5, COEF)
        rval, cval, t, cne, rne = game.DO_Nash(g, EP, 0, "none")

        means[i] = iters.mean()
        stds[i] = iters.std()
        means_nop[i] = t

    plt.plot(SIZES, means)
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    plt.plot(SIZES, means_nop)
    plt.show()

    data = np.array([means, stds, means_nop])
    df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds', 'means_nop'])
    df.to_csv("experiments/SDO_path_planning.csv", mode='a')