import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import zero_sum as game

##########################################################################################################
# Matrix examples
##########################################################################################################

def rand_sym_game(n):
    M = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            M[i,j] = np.random.uniform(-1, 1) #np.random.randint(0,2)
    return M - np.transpose(M)

def rand_game(m, n):
    M = np.random.uniform(0,1,(m,n))
    return M

def ex1(n):
    M = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            if j==i+1:
                M[i,j] = 2
            else:
                M[i,j] = 1
    return M - np.transpose(M)

def ex2(n):
    M = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            M[i,j] = 1
    return M - np.transpose(M)

def exU(n):
    M = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            if j==i+1:
                M[i,j] = -0.5
            else:
                M[i,j] = -0.25 #-1.1*EP
    return ((M - np.transpose(M))+0.5)    

def exS(n):
    M = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            if j==i+1:
                M[i,j] = -1
            else:
                M[i,j] = -1 #-1.1*EP
    return (M - np.transpose(M))    

def exMorra(n):
    M = np.zeros((n**2,n**2))
    pairs = [(i,j) for i in range(n) for j in range(n)]
    for (i1,j1) in pairs:
        for (i2,j2) in pairs:
            if j1==i2 and j2!=i1:
                M[i1*n+j1,i2*n+j2] = -(i1+i2)
            elif j1!=i2 and j2==i1:
                M[i1*n+j1,i2*n+j2] = (i1+i2)
            else:
                M[i1*n+j1,i2*n+j2] = 0
    return (1/(2*n-2))*M

def exMorra_crisp(n):
    M = np.zeros((n**2,n**2))
    pairs = [(i,j) for i in range(n) for j in range(n)]
    for (i1,j1) in pairs:
        for (i2,j2) in pairs:
            if j1==i2 and j2!=i1:
                M[i1*n+j1,i2*n+j2] = -1
            elif j1!=i2 and j2==i1:
                M[i1*n+j1,i2*n+j2] = 1
            else:
                M[i1*n+j1,i2*n+j2] = 0
    return M

###################################################################################################
# Examples efficient perturbations
###################################################################################################
class Game(ABC):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.num_terminals = self.get_num_of_terminals()
        self.M = self.build_matrix()
        self.masks = self.build_masks()
    
    @abstractmethod
    def build_matrix(self) -> npt.NDArray:
        pass

    @abstractmethod
    def get_num_of_terminals(self) -> int:
        pass

    @abstractmethod
    def get_terminal_index(self, x, y) -> int:
        pass

    def perturb_matrix(self, param, type):
        noise = get_noise(param, type, len(self.masks))
        perbM = np.zeros((self.m, self.n))
        for i in range(len(self.masks)):
            perbM += noise[i]*self.masks[i]
        return self.M + perbM 

    def build_masks(self):
        masks = np.zeros((self.num_terminals, self.n, self.n))
        masks[0] = np.eye(self.n)
        for i in range(self.n):
            for j in range(self.n):
                ind = self.get_terminal_index(i, j)
                masks[ind, i, j] = 1
        return masks

class Theorem3_3(Game):
    def __init__(self, n):
        super().__init__(n,n)

    def get_num_of_terminals(self):
        self.num_bits = int(np.ceil(np.log2(self.n)))
        return 1 + 2*self.num_bits
    
    def build_matrix(self):
        M = np.zeros((self.n,self.n))
        for i in range(0,self.n):
            for j in range(i+1,self.n):
                M[i,j] = 1
        return (M - np.transpose(M))  
        
    def get_terminal_index(self, x, y):
        xb = np.binary_repr(x, width=self.num_bits)
        yb = np.binary_repr(y, width=self.num_bits)
        out = 0
        for i in range(len(xb)):
            if xb[i] > yb[i]:
                out = i + 1
                break
            elif xb[i] < yb[i]:
                out = self.num_bits + i + 1
                break
        return out

class Theorem3_2(Game):
    def __init__(self, n):
        super().__init__(n,n)

    def get_num_of_terminals(self):
        self.num_bits = int(np.ceil(np.log2(self.n)))
        return 1 + 2*self.num_bits

    def build_matrix(self):
        M = np.zeros((self.n,self.n))
        for i in range(0,self.n):
            for j in range(i+1,self.n):
                if j==i+1:
                    M[i,j] = 2
                else:
                    M[i,j] = 1
        return (M - np.transpose(M))  
       
    def get_terminal_index(self, x, y):
        xb = np.binary_repr(x, width=self.num_bits)
        yb = np.binary_repr(y, width=self.num_bits)
        out = 0
        if x-y == 1:
            out = self.num_bits
        elif y-x == 1:
            out = 2*self.num_bits
        else:
            for i in range(len(xb)):
                if xb[i] > yb[i]:
                    out = i + 1
                    break
                elif xb[i] < yb[i]:
                    out = self.num_bits + i + 1
                    break
        return out

class Theorem3_5(Game):
    def __init__(self, n):
        self.M1 = np.array([[1,-1],[-1,1]])
        self.M2 = np.array([[1,-1],[1,1]])
        super().__init__(n,n)

    def get_num_of_terminals(self):
        self.num_bits = int(np.ceil(np.log2(self.n)))
        return 4*self.num_bits

    def build_matrix(self):
        M = np.zeros((self.n,self.n))
        for x in range(0,self.n):
            for y in range(0,self.n):
                xb = np.binary_repr(x, width=self.num_bits)
                yb = np.binary_repr(y, width=self.num_bits)
                outcomes = np.zeros(self.num_bits)
                for i in range(self.num_bits):
                    xi = int(xb[i])
                    yi = int(yb[i])
                    if i == 0:
                        outcomes[i] = self.M1[xi,yi]
                    else:
                        outcomes[i] = self.M2[xi,yi]
                # print(x,y,outcomes)
                if x == 0:
                    M[-1,y] = outcomes.mean()
                elif x == self.n-1: 
                    M[0,y] = outcomes.mean()
                else:
                    M[x,y] = outcomes.mean()
        return -M

    def get_terminal_index(self, x, y):
        return 0
    
###############################################################################
# General utilities
###############################################################################

def get_noise(param, type, size):
    match type:
        case "gumbel":
            noise = np.random.gumbel(0, param, size)
        case "uniform":
            noise = np.random.uniform(-param, param, size)  
        case "normal":
            noise = np.random.normal(0, param, size)
        case _:
            noise = np.zeros(size)
    return noise

def update_dict(dict, key, val):
    if key in dict:
        dict[key] += val
    else:
        dict[key] = val
    return dict

def normalize_dict(dict):
    s = sum(dict.values())
    return { k:(v/s) for (k,v) in dict.items() }

def get_vals(M, strategy):
    keys = list(strategy.keys())
    ps = np.array(list(strategy.values()))
    vals = np.matmul(ps, M[keys])
    return vals

def build_sub_matrix(g, col_strategy, row_strategy):
    rkeys = sorted(row_strategy.keys())
    ckeys = sorted(col_strategy.keys())
    M2 = g.M[rkeys]
    return M2[:,ckeys], ckeys, rkeys

def best_row_response_index(g, col_mixed_strategy, param, type):
    M = g.perturb_matrix(param, type)
    vals = get_vals(np.transpose(M), col_mixed_strategy)
    index = np.argmin(vals)
    return index

def best_row_response_val(g, col_mixed_strategy):
    vals = get_vals(np.transpose(g.M), col_mixed_strategy)
    m = np.min(vals)
    return m

def best_col_response_index(g, row_mixed_strategy, param, type):
    M = g.perturb_matrix(param, type)
    vals = get_vals(M, row_mixed_strategy)
    index = np.argmax(vals)
    return index

def best_col_response_val(g, row_mixed_strategy):
    vals = get_vals(g.M, row_mixed_strategy)
    m = np.max(vals)
    return m

def DO_Nash(g, ep, param=0.3, type="normal"):
    t = 1

    k = 0 #np.random.randint(0, len(g.M))
    l = 0 #np.random.randint(0, len(g.M))
    X = {k:1.0}  
    Y = {l:1.0}

    cval = best_col_response_val(g, Y)
    rval = best_row_response_val(g, X)
    
    while cval-rval > ep:
        # print(X,Y)
        # if t % 100 == 0:
        #     print(t)
        t += 1
        l = best_row_response_index(g, X, param, type)
        k = best_col_response_index(g, Y, param, type)
        Y = update_dict(Y, l, 1)
        X = update_dict(X, k, 1)

        # DO update
        S, ckeys, rkeys = build_sub_matrix(g, X, Y)
        _, cstrag, rstrag = game.solve_game(np.transpose(-S))
        X = { k:cstrag[i] for (k,i) in zip(ckeys, range(len(cstrag))) }
        Y = { k:rstrag[i] for (k,i) in zip(rkeys, range(len(rstrag))) }
        rval = best_row_response_val(g, X)
        cval = best_col_response_val(g, Y)

    return rval, cval, t, normalize_dict(X), normalize_dict(Y)

if __name__ == "__main__":

    g = Theorem3_3(256)
    print(g.M)
    # print(g.masks)
    rval, cval, t, cne, rne = DO_Nash(g, 0.1, 0, "uniform")
    print(cval, rval, t, cne, rne)
