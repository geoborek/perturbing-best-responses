import examples
import perturb_FP as solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

EP = 0.1
TYPE = "gumbel"
# PARAM = 2/EP

SIZES = range(5, 106, 10) 
REPETITIONS = 10

np.random.seed(1)

if __name__ == "__main__":
    
    means = np.zeros(len(SIZES))
    stds = np.zeros(len(SIZES))
    means_nop = np.zeros(len(SIZES))
    means_afp = np.zeros(len(SIZES))
    means_afp_p = np.zeros(len(SIZES))
    stds_afp_p = np.zeros(len(SIZES))

    for i, n in tqdm(enumerate(SIZES)):

        sqrtT = (2+np.sqrt(2*np.log(n**2)))/EP
        PARAM = sqrtT/np.sqrt(8*np.log(n**2))
        # print(PARAM)

        iters = np.zeros(REPETITIONS)
        iters_afp_p = np.zeros(REPETITIONS)
        M = examples.exMorra(n)/2 + 0.5
        for r in range(REPETITIONS):
            # g = examples.Theorem3_3(n)
            rval, cval, t, cne, rne = solver.FP_Nash(M, EP, PARAM, TYPE)
            # print(rval, cval, t)
            iters[r] = t
            rval, cval, t, cne, rne = solver.AFP_Nash(M, EP, PARAM, TYPE)
            iters_afp_p[r] = t
        means[i] = iters.mean()
        stds[i] = iters.std()
        means_afp_p[i] = iters_afp_p.mean()
        stds_afp_p[i] = iters_afp_p.std()

        rval, cval, t, cne, rne = solver.FP_Nash(M, EP, 0, "none")
        means_nop[i] = t
        rval, cval, t, cne, rne = solver.AFP_Nash(M, EP, 0, "none")
        means_afp[i] = t

    plt.rcParams.update({'font.size': 20})
    plt.plot(SIZES, means, label="SFP")
    plt.fill_between(SIZES, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    plt.plot(SIZES, means_nop, label="FP")
    plt.plot(SIZES, means_afp, label="AFP")
    plt.plot(SIZES, means_afp_p, label="SAFP")
    plt.fill_between(SIZES, np.subtract(means_afp_p, stds_afp_p), np.add(means_afp_p, stds_afp_p), alpha=0.2)
    plt.xlabel("size")
    plt.ylabel("iterations")
    plt.legend(loc="upper left")
    plt.show()

    data = np.array([means, stds, means_nop, means_afp, means_afp_p, stds_afp_p])
    df = pd.DataFrame(data, columns=SIZES, index=['means', 'stds', 'means_nop', 'means_afp', 'means_afp_p', 'stds_afp_p'])
    df.to_csv("experiments/SFP_Morra.csv", mode='a')