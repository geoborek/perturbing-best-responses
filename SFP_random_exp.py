import examples
import perturb_FP as solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

EP = 0.1
TYPE = "gumbel"

SIZES = range(100, 1001, 100) 
REPETITIONS = 100

np.random.seed(1)

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))
    stds_nop = np.zeros(len(SIZES))
    means_afp = np.zeros(len(SIZES))
    stds_afp = np.zeros(len(SIZES))
    means_afp_p = np.zeros(len(SIZES))
    stds_afp_p = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):

        sqrtT = (2+np.sqrt(2*np.log(n)))/EP
        PARAM = sqrtT/np.sqrt(8*np.log(n)) 
        # print(PARAM)

        iters = np.zeros(REPETITIONS)
        iters_nop = np.zeros(REPETITIONS)
        iters_afp = np.zeros(REPETITIONS)
        iters_afp_p = np.zeros(REPETITIONS)
        for r in range(REPETITIONS):
            # g = examples.Theorem3_3(n)
            M = examples.rand_game(n,n)
            rval, cval, t, cne, rne = solver.FP_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t
            rval, cval, t, cne, rne = solver.FP_Nash(M, EP, 0, "none")
            iters_nop[r] = t
            rval, cval, t, cne, rne = solver.AFP_Nash(M, EP, 0, "none")
            iters_afp[r] = t
            rval, cval, t, cne, rne = solver.AFP_Nash(M, EP, PARAM, TYPE)
            iters_afp_p[r] = t

        means[i] = iters.mean()
        stds[i] = iters.std()
        means_nop[i] = iters_nop.mean()
        stds_nop[i] = iters_nop.std()
        means_afp[i] = iters_afp.mean()
        stds_afp[i] = iters_afp.std()
        means_afp_p[i] = iters_afp_p.mean()
        stds_afp_p[i] = iters_afp_p.std()

    plt.plot(SIZES, means, label="SFP")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    plt.plot(SIZES, means_nop, label="FP")
    plt.fill_between(SIZES, np.subtract(means_nop, stds_nop), np.add(means_nop, stds_nop), alpha=0.2)
    plt.plot(SIZES, means_afp, label="AFP")
    plt.fill_between(SIZES, np.subtract(means_afp, stds_afp), np.add(means_afp, stds_afp), alpha=0.2)
    plt.plot(SIZES, means_afp_p, label="SAFP")
    plt.fill_between(SIZES, np.subtract(means_afp_p, stds_afp_p), np.add(means_afp_p, stds_afp_p), alpha=0.2)
    # plt.title("gumbel perturbation")
    plt.xlabel("size")
    plt.ylabel("iterations")
    plt.legend(loc="upper left")
    plt.show()

    data = np.array([means, stds, means_nop, stds_nop, means_afp, stds_afp, means_afp_p, stds_afp_p])
    df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds', 'means_nop', 'stds_nop', 'means_afp', 'stds_afp', 'means_afp_p', 'stds_afp_p'])
    df.to_csv("experiments/SFP_random.csv", mode='a')