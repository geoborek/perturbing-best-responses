import examples
import perturb_FP as solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

EP = 0.1
TYPE = "uniform"
PARAM = 1

SIZES = range(50, 1001, 25) 
REPETITIONS = 100

np.random.seed(1)

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))
    stds_nop = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):

        # print(PARAM)

        iters = np.zeros(REPETITIONS)
        iters_nop = np.zeros(REPETITIONS)
        M = examples.ex1(n)
        for r in range(REPETITIONS):
            # g = examples.Theorem3_3(n)
            rval, cval, t, cne, rne = solver.DO_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t

        means[i] = iters.mean()
        stds[i] = iters.std()

    plt.rcParams.update({'font.size': 20})
    plt.plot(SIZES, means, label="U")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    # plt.plot(SIZES, means_nop, label="FP")

    for i, n in tqdm(enumerate(SIZES)):

        # print(PARAM)

        iters = np.zeros(REPETITIONS)
        iters_nop = np.zeros(REPETITIONS)
        M = examples.ex2(n)
        for r in range(REPETITIONS):
            # g = examples.Theorem3_3(n)
            rval, cval, t, cne, rne = solver.DO_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t

        means[i] = iters.mean()
        stds[i] = iters.std()

    plt.plot(SIZES, means, label="L")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)

    plt.title("uniform perturbation")
    plt.xlabel("size")
    plt.ylabel("iterations")
    plt.legend(loc="upper left")
    plt.show()

    # data = np.array([means, stds])
    # df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds'])
    # df.to_csv("experiments/SDO_3_2.csv", mode='a')