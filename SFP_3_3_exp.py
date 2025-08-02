import examples
import perturb_FP as solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

EP = 0.1
TYPE = "gumbel"
# PARAM = 2/EP

SIZES = range(100, 1001, 100) 
REPETITIONS = 10

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))
    stds_nop = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):

        sqrtT = (2+np.sqrt(2*np.log(n)))/EP
        PARAM = sqrtT/np.sqrt(8*np.log(n)) 
        # print(PARAM)

        iters = np.zeros(REPETITIONS)
        iters_nop = np.zeros(REPETITIONS)
        M = -examples.ex3_3(n)
        for r in range(REPETITIONS):
            rval, cval, t, cne, rne = solver.FP_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t
        rval, cval, t, cne, rne = solver.FP_Nash(M, EP, 0, "none")

        means[i] = iters.mean()
        stds[i] = iters.std()
        means_nop[i] = t

    plt.plot(SIZES, means, label="SFP")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    plt.plot(SIZES, means_nop, label="FP")
    plt.xlabel("size")
    plt.ylabel("iterations")
    plt.legend(loc="upper left")

    plt.show()

    data = np.array([means, stds, means_nop])
    df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds', 'means_nop'])
    df.to_csv("experiments/SFP_3_3.csv", mode='a')