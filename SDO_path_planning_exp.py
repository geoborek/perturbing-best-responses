import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import game

EP = 0.1
TYPE = "uniform"
PARAM = 0.01
COEF = 10

SIZES = range(3, 15) 
REPETITIONS = 10

np.random.seed(1)

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):
        
        iters = np.zeros(REPETITIONS)
        for r in range(REPETITIONS):
            g = game.Grid(n-1, COEF)
            rval, cval, t, cne, rne = game.DO_Nash(g, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t
        g = game.Grid(n-1, COEF)
        rval, cval, t, cne, rne = game.DO_Nash(g, EP, 0, "none")

        means[i] = iters.mean()
        stds[i] = iters.std()
        means_nop[i] = t

    plt.rcParams.update({'font.size': 20})
    plt.plot(SIZES, means_nop, label="No pert.")
    plt.plot(SIZES, means, label="d=0.01")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)

    PARAM = 0.001
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):
        
        iters = np.zeros(REPETITIONS)
        for r in range(REPETITIONS):
            g = game.Grid(n-1, COEF)
            rval, cval, t, cne, rne = game.DO_Nash(g, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t

        means[i] = iters.mean()
        stds[i] = iters.std()

    plt.plot(SIZES, means, label="d=0.001")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)


    plt.title("uniform perturbation")
    plt.xlabel("grid size")
    plt.ylabel("iterations")
    plt.legend(loc="upper left")

    plt.show()

    # data = np.array([means, stds, means_nop])
    # df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds', 'means_nop'])
    # df.to_csv("experiments/SDO_path_planning.csv", mode='a')