import numpy as np
import zero_sum as game
import examples
import matplotlib.pyplot as plt
import perturb_FP as do
import examples as ex

EP = 0.1

def get_uncovered(M, X, lb):
    m = len(M)      # num of rows
    n = len(M[0])   # num of cols
    
    s = np.zeros(m)
    for i, v in X.items():
        s += v*np.transpose(M)[i]

    return {k:1 for k in range(m) if s[k] == lb}

def set_cover(M, c):
    t = 1
    m = len(M)      # num of rows

    Y = {k:1 for k in range(m)}
    uc = sum(Y.values())

    k = do.best_col_response_index(M, Y, 0, "none")
    X = {k:1}
    lb = do.best_row_response_val(M, X)
    
    while lb < c:
        t += 1
        Y = get_uncovered(M, X, lb)
        nuc = sum(Y.values())
        if nuc == uc:
            print(f"{c}-cover does not exist.")
            break
        else:
            uc = nuc
        # print(Y)     
        k = do.best_col_response_index(M, Y, 0, "none")
        X = do.update_dict(X, k, 1)
        lb = do.best_row_response_val(M, X)
        # print(lb)
    return X, c/t

def DO_Nash(M, ep, X, Y, param=0.0, type="none"):
    t = 1

    cval = do.best_col_response_val(M, Y)
    rval = do.best_row_response_val(M, X)
    
    while cval-rval > ep:
        # print(X,Y)
        # if t % 100 == 0:
        #     print(t)
        t += 1
        l = do.best_row_response_index(M, X, param, type)
        k = do.best_col_response_index(M, Y, param, type)
        Y = do.update_dict(Y, l, 1)
        X = do.update_dict(X, k, 1)

        # DO update
        S, ckeys, rkeys = do.build_sub_matrix(M, X, Y)
        # _, _, n, cstrag, rstrag = FP_Nash(S, 0.05)
        _, cstrag, rstrag = game.solve_game(np.transpose(-S))
        X = { k:cstrag[i] for (k,i) in zip(ckeys, range(len(cstrag))) }
        Y = { k:rstrag[i] for (k,i) in zip(rkeys, range(len(rstrag))) }
        # print(X,Y)
        rval = do.best_row_response_val(M, X)
        cval = do.best_col_response_val(M, Y)
        # print(cval, rval)

    return rval, cval, t, do.normalize_dict(X), do.normalize_dict(Y)

if __name__ == "__main__":

    M = np.array([[1,0,1,0], 
                  [1,1,0,0], 
                  [0,1,1,0], 
                  [0,0,0,1]])

    M = np.array([[1,0,0,0,1,0],
                  [1,0,0,0,0,1],
                  [1,0,0,0,1,0],
                  [1,0,0,0,0,1], 
                  [0,1,0,0,1,0], 
                  [0,1,0,0,0,1], 
                  [0,0,1,0,1,0], 
                  [0,0,0,1,0,1]])

    M = 1*np.random.randint(0,2,(200,200))

    # val, _, _ = game.solve_game(np.transpose(-M))
    rval, cval, t, cne, rne = DO_Nash(M, EP, {0:1.0}, {0:1.0}, 0, "none")
    print(f"Game value upper bound = {cval}")
    print(f"Game value lower bound = {rval}")
    print(f"Number of iterations = {t}")

    v = 0
    c = 1
    X, nv = set_cover(M,c)
    while nv - v > EP:
        v = nv   
        c += 1     
        X, nv = set_cover(M,c)
    print(f"Column set {c}-cover: {X}")
    print(f"Lower bound = {v}")

    MT = np.transpose(M)
    v = 0
    c = 1
    Y, nv = set_cover(np.ones(MT.shape)-MT,c)
    while nv - v > EP:
        v = nv   
        c += 1     
        Y, nv = set_cover(np.ones(MT.shape)-MT,c)
    # print(np.ones(MT.shape)-MT)
    print(f"Row set {c}-cover: {Y}")
    print(f"Upper bound = {1-v}")

    rval, cval, t, cne, rne = DO_Nash(M, EP, do.normalize_dict(X), do.normalize_dict(Y), 0, "none")
    print(f"Game value upper bound = {cval}")
    print(f"Game value lower bound = {rval}")
    print(f"Number of iterations = {t}")
