import numpy as np
import zero_sum as game
import examples
import matplotlib.pyplot as plt

EP = 0.1
SIGMA = 0.1

def softmax(scores, eta):
    es = np.exp(eta*scores)
    return es/np.sum(es)

def expected_index(scores, eta):
    distr = softmax(scores, eta)
    index = 0
    for i, p in enumerate(distr):
        index += i*p
    return index

def expected_perturb_index(scores, num_simulatons):
    indexes = np.zeros(num_simulatons)
    for i in range(num_simulatons):
        # perturb_scores = scores + np.abs(np.random.normal(0, SIGMA, len(scores)))  # perturbation
        perturb_scores = perturb(scores, 0.5, "uniform")
        indexes[i] = np.argmax(perturb_scores)
    return np.mean(indexes)

def update_dict(dict, key, val):
    if key in dict:
        dict[key] += val
    else:
        dict[key] = val
    return dict

def normalize_dict(dict):
    s = sum(dict.values())
    return { k:(v/s) for (k,v) in dict.items() }

def perturb(vals, param, type="normal", op="add"):
    match type:
        case "gumbel":
            noise = np.random.gumbel(0, param, len(vals))
        case "uniform":
            noise = np.random.uniform(-param, param, len(vals))  
        case "normal":
            noise = np.random.normal(0, param, len(vals))
        case _:
            noise = np.zeros(len(vals))
    if op == "add":
        return vals+noise
    else:
        return vals-noise

def get_vals(M, strategy):
    keys = list(strategy.keys())
    ps = np.array(list(strategy.values()))
    vals = np.matmul(ps, M[keys])
    return vals

def build_sub_matrix(M, col_strategy, row_strategy):
    rkeys = sorted(row_strategy.keys())
    ckeys = sorted(col_strategy.keys())
    M2 = M[rkeys]
    return M2[:,ckeys], ckeys, rkeys

def best_row_response_index(M, col_mixed_strategy, param, type):
    vals = get_vals(np.transpose(M), col_mixed_strategy)
    index = np.argmin(perturb(vals, param, type, "sub"))
    return index

def best_row_response_val(M, col_mixed_strategy):
    vals = get_vals(np.transpose(M), col_mixed_strategy)
    m = np.min(vals)
    return m

def best_col_response_index(M, row_mixed_strategy, param, type):
    vals = get_vals(M, row_mixed_strategy)
    index = np.argmax(perturb(vals, param, type))
    return index

def best_col_response_val(M, row_mixed_strategy):
    vals = get_vals(M, row_mixed_strategy)
    m = np.max(vals)
    return m

def FP_Nash(M, ep, param, type="gumbel"):
    t = 1

    k = 0 #np.random.randint(0, M.shape[1])
    l = 0 #np.random.randint(0, M.shape[0])
    X = {k:1}  
    Y = {l:1}

    cval = best_col_response_val(M, Y)
    rval = best_row_response_val(M, X)
    
    while cval-rval > t*ep:

        # if t % 100 == 0:
        #     print(t)
        t += 1
        # lc = best_row_response_index(M, X, 0, "none")
        # kc = best_col_response_index(M, Y, 0, "none")
        l = best_row_response_index(M, X, param, type)
        k = best_col_response_index(M, Y, param, type)

        # FP update
        Y = update_dict(Y, l, 1)
        X = update_dict(X, k, 1)

        rval = best_row_response_val(M, X)
        cval = best_col_response_val(M, Y)

    return rval/t, cval/t, t, normalize_dict(X), normalize_dict(Y)

def AFP_Nash(M, ep, param=0.0, type="gumbel"):
    t = 1

    k = 0 #np.random.randint(0, M.shape[1])
    l = 0 #np.random.randint(0, M.shape[0])
    X = {k:1}  
    Y = {l:1}

    cval = best_col_response_val(M, Y)
    rval = best_row_response_val(M, X)
    
    while cval-rval > t*ep:
        # print((cval-rval)/t)
        # if t % 100 == 0:
        #     print(t)
        t += 1
        l = best_row_response_index(M, X, param, type)
        k = best_col_response_index(M, Y, param, type)

        # FP update
        Yp = update_dict(Y.copy(), l, 1)
        Xp = update_dict(X.copy(), k, 1)

        l = best_row_response_index(M, Xp, param, type)
        k = best_col_response_index(M, Yp, param, type)

        # FP update
        Y = update_dict(Y, l, 1)
        X = update_dict(X, k, 1)

        rval = best_row_response_val(M, X)
        cval = best_col_response_val(M, Y)

    return rval/t, cval/t, t, normalize_dict(X), normalize_dict(Y)

def DO_Nash(M, ep, param=0.0, type="none"):
    t = 1

    k = 0 #np.random.randint(0, len(M))
    l = 0 #np.random.randint(0, len(M))
    X = {k:1.0}  
    Y = {l:1.0}

    cval = best_col_response_val(M, Y)
    rval = best_row_response_val(M, X)
    
    while cval-rval > ep:
        # print(cval-rval)
        # print(X,Y)
        # if t % 100 == 0:
        #     print(t)
        t += 1
        l = best_row_response_index(M, X, param, type)
        k = best_col_response_index(M, Y, param, type)
        Y = update_dict(Y, l, 1)
        X = update_dict(X, k, 1)


        # DO update
        S, ckeys, rkeys = build_sub_matrix(M, X, Y)
        # _, _, n, cstrag, rstrag = FP_Nash(S, 0.05)
        _, cstrag, rstrag = game.solve_game(np.transpose(-S))

        X = { k:cstrag[i] for (k,i) in zip(ckeys, range(len(cstrag))) }
        Y = { k:rstrag[i] for (k,i) in zip(rkeys, range(len(rstrag))) }
        # print(X,Y)
        rval = best_row_response_val(M, X)
        cval = best_col_response_val(M, Y)
        # print(cval, rval)

    return rval, cval, t, normalize_dict(X), normalize_dict(Y)

if __name__ == "__main__":

    M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    M = 1*np.random.randint(0,2,(50,50))
    # M[:,-1] = 0.6*np.ones(400)
    # M = -gm.ex3_2(8)
    # M = np.array(
    #     [
    #         [  1,  1/3,  1/3, -1/3, 1/3, -1/3, -1/3, -1],
    #         [  1,    1,  1/3,  1/3, 1/3,  1/3, -1/3, -1/3],
    #         [  1,  1/3,    1,  1/3, 1/3, -1/3,  1/3, -1/3],
    #         [  1,    1,    1,    1, 1/3,  1/3,  1/3,  1/3],
    #         [1/3, -1/3, -1/3,   -1,   1,  1/3,  1/3, -1/3],
    #         [1/3,  1/3, -1/3, -1/3,   1,    1,  1/3,  1/3],
    #         [1/3, -1/3,  1/3, -1/3,   1,  1/3,    1,  1/3],
    #         [1/3,  1/3,  1/3,  1/3,   1,    1,    1,    1]
    #     ]
    # )
    # M = gm.symmetric_game(M)
    # M = gm.rand_sym_game(500)
    n = 40
    M = examples.exMorra(n)
    # M = np.load("theorem_3.5.npy")
    # M = examples.rand_game(100, 100)
    # print(M)
    rval, cval, t, cne, rne = AFP_Nash(M, EP, 2/EP, "gumbel")
    print(cval, rval, t)# , cne, rne)

    rval, cval, t, cne, rne = DO_Nash(M, EP, SIGMA, "normal")
    print(cval, rval, t) #, cne, rne)

