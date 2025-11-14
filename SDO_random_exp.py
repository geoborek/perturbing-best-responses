import examples
import perturb_FP as solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

EP = 0.1
TYPE = "uniform"
PARAM = 0.01

SIZES = range(10, 501, 10) 
REPETITIONS = 10

np.random.seed(1)

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))
    stds_nop = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):
        
        iters = np.zeros(REPETITIONS)
        iters_nop = np.zeros(REPETITIONS)
        for r in range(REPETITIONS):
            # g = examples.Theorem3_3(n)
            M = examples.rand_game(n, n)
            rval, cval, t, cne, rne = solver.DO_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t
            rval, cval, t, cne, rne = solver.DO_Nash(M, EP, 0, "none")
            iters_nop[r] = t

        means[i] = iters.mean()
        stds[i] = iters.std()
        means_nop[i] = iters_nop.mean()
        stds_nop[i] = iters_nop.std()

    plt.plot(SIZES, means, label="U(-0.01,0.01)")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    plt.plot(SIZES, means_nop, label="No perturbation")
    plt.fill_between(SIZES, np.subtract(means_nop, stds_nop), np.add(means_nop, stds_nop), alpha=0.2)

    PARAM = 0.1
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):
        
        iters = np.zeros(REPETITIONS)
        for r in range(REPETITIONS):
            # g = examples.Theorem3_3(n)
            M = examples.rand_game(n, n)
            rval, cval, t, cne, rne = solver.DO_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t

        means[i] = iters.mean()
        stds[i] = iters.std()

    plt.plot(SIZES, means, label="U(-0.1,0.1)")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)


    plt.title("uniform perturbation")
    plt.xlabel("size")
    plt.ylabel("iterations")
    plt.legend(loc="upper left")

    plt.show()

    # data = np.array([means, stds, means_nop, stds_nop])
    # df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds', 'means_nop', 'stds_nop'])
    # df.to_csv("experiments/SDO_random.csv", mode='a')