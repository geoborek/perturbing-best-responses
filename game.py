from abc import ABC, abstractmethod
from functools import partial
import astar
import random
import numpy as np
import zero_sum as zs
import perturb_FP as fp

def modify_vector(x, i, coef):
    mask = np.ones(x.shape)
    mask[i] = coef
    return np.multiply(x, mask)

def distance(n, coef, x, y):
    f1 = (x + y) / n
    f2 = (2*n - x - y) / n
    return min(f1, f2) #/(2*n-1+coef)

class Game(ABC):

    def __init__(self):
        self.num1 = 0
        self.num2 = 0
        self.strategies1 = {}
        self.strategies2 = {}
        self.M = np.empty
    
    def add_strategy1(self, strag):
        self.strategies1[self.num1] = strag
        self.num1 += 1

        if self.num1 == 1 and self.num2 == 1:
            self.M = np.array([[self.get_utility(strag, self.strategies2[0])]])
        elif self.num2 > 0:
            vals = [ self.get_utility(strag, self.strategies2[i]) for i in range(self.num2) ]
            # print(self.M, vals)
            self.M = np.vstack((self.M, vals))

    def add_strategy2(self, strag):
        self.strategies2[self.num2] = strag
        self.num2 += 1

        if self.num1 == 1 and self.num2 == 1:
            self.M = np.array([[self.get_utility(self.strategies1[0], strag)]])
        elif self.num1 > 0:
            vals = [ [self.get_utility(self.strategies1[i], strag)] for i in range(self.num1) ]
            self.M = np.hstack((self.M, vals))

    @abstractmethod
    def get_utility(self, strag1, strag2):
        pass

    @abstractmethod
    def get_random_strategy1(self):
        pass

    @abstractmethod
    def get_random_strategy2(self):
        pass

    # @abstractmethod
    # def best_response1_val(self, mixed_strag):
    #     pass

    # @abstractmethod
    # def best_response_strag(self, player, mixed_strag):
    #     pass

class Grid(Game):

    def __init__(self, n, coef):
        super().__init__()
        self.size = n
        self.coef = coef
        self.costs, self.actions = self.generate_costs()
        self.s0 = (0,0)

    def generate_costs(self):
        actions = {}
        costs = np.zeros(2*(self.size+1)*self.size)
        k = 0
        for i in range(self.size+1):
            for j in range(self.size+1):
                if i != self.size:
                    actions[((i,j),0)] = k
                    costs[k] = distance(self.size, self.coef, i+0.5, j) #(self.size - abs(self.size-i-j))/(3*self.size) 
                    k += 1
                if j != self.size:
                    actions[((i,j),1)] = k
                    costs[k] = distance(self.size, self.coef, i, j+0.5) #(self.size - abs(self.size-i-j))/(3*self.size) 
                    k += 1
        # print(actions)
        # print(costs)
        return costs, actions

    def get_applicable(self, s):
        out = []
        if s[0] < self.size:
            out.append((s,0))
        if s[1] < self.size:
            out.append((s,1))
        return out

    def succ(self, a, s):
        if a[1] == 0:
            return (s[0]+1,s[1])
        else:
            return (s[0],s[1]+1)

    def is_goal(self, s):
        return s[0] == self.size and s[1] == self.size

    def heur(self, s):
        return 0

    def get_utility(self, plan, action):
        costs = modify_vector(self.costs, self.actions[action], self.coef)
        return self.get_plan_cost(costs, plan)
        
    def cost(self, costs, a):
        return costs[self.actions[a]]
    
    def get_plan_cost(self, costs, plan):
        return sum([self.cost(costs, a) for a in plan])
    
    def get_random_strategy1(self):
        plan = astar.astar(self.s0, self.is_goal, self.get_applicable, self.succ, partial(self.cost, self.costs), self.heur)
        self.add_strategy1(plan)
    
    def get_random_strategy2(self):
        actions = list(self.actions.keys())
        i = 0 #random.randint(0, len(actions)-1)
        self.add_strategy2(actions[i])
            
    def best_response1(self, mixed_strag, param, type):
        costs = fp.perturb(self.costs, param, type)

        M = np.zeros((len(mixed_strag), len(self.costs)))
        for i in range(len(mixed_strag)):
            M[i] = modify_vector(costs, self.actions[self.strategies2[i]], self.coef)
        
        costs = np.matmul(mixed_strag, M)
        plan = astar.astar(self.s0, self.is_goal, self.get_applicable, self.succ, partial(self.cost, costs), self.heur)
        return plan, costs

    def best_response1_val(self, mixed_strag):
        plan, costs = self.best_response1(mixed_strag, 0, "none")
        return self.get_plan_cost(costs, plan)
    
    def get_mixed_plan_cost(self, costs, mixed_strag):
        c = 0
        for i in range(len(mixed_strag)):
            p = mixed_strag[i]
            if p > 0:
                plan = self.strategies1[i]
                c += p*self.get_plan_cost(costs, plan)
        return c

    def best_response2(self, mixed_strag, param, type):
        costs = fp.perturb(self.costs, param, type)

        m = -np.inf
        for a, i in self.actions.items():
            cs = modify_vector(costs, i, self.coef)
            c = self.get_mixed_plan_cost(cs, mixed_strag)
            # print(a, cs)
            if c > m:
                m = c
                best_action = a 
        return best_action
    
    def best_response2_val(self, mixed_strag):
        action = self.best_response2(mixed_strag, 0, "none")
        costs = modify_vector(self.costs, self.actions[action], self.coef)
        return self.get_mixed_plan_cost(costs, mixed_strag)

def DO_Nash(g, ep, param, type="normal"):
    t = 1

    g.get_random_strategy1()
    g.get_random_strategy2()
    X = [1.0]  
    Y = [1.0]

    rval = g.best_response1_val(X)
    cval = g.best_response2_val(Y)
    
    while cval-rval > ep:
        # print(X,Y)
        # if t % 10 == 0:
        #     print(t)
        t += 1
        plan, costs = g.best_response1(X, param, type)
        action = g.best_response2(Y, param, type)
        if plan not in g.strategies1.values():
            g.add_strategy1(plan)
        if action not in g.strategies2.values():
            g.add_strategy2(action)

        # print(g.M)
        # DO update
        _, X, Y = zs.solve_game(np.transpose(-g.M))

        # print(X,Y)
        rval = g.best_response1_val(X)
        cval = g.best_response2_val(Y)
        # print(cval, rval)

    return rval, cval, t, X, Y

def FP_Nash(g, ep, param, type="gumbel"):
    t = 1

    g.get_random_strategy1()
    g.get_random_strategy2()
    X = [1.0]  
    Y = [1.0]

    rval = g.best_response1_val(X)
    cval = g.best_response2_val(Y)
    
    while cval-rval > t*ep:

        # if t % 100 == 0:
        #     print(t)
        t += 1
        plan, costs = g.best_response1(X, param, type)
        action = g.best_response2(Y, param, type)

        # FP update
        if plan not in g.strategies1.values():
            g.add_strategy1(plan)
            Y.append(1.0)
        else:
            index = list(g.strategies1.keys())[list(g.strategies1.values()).index(plan)]
            Y[index] = Y[index] + 1
        if action not in g.strategies2.values():
            g.add_strategy2(action)
            X.append(1.0)
        else:
            index = list(g.strategies2.keys())[list(g.strategies2.values()).index(action)]
            X[index] = X[index] + 1

        # Y = update_dict(Y, l, 1)
        # X = update_dict(X, k, 1)

        rval = g.best_response1_val(X)
        cval = g.best_response2_val(Y)

    return rval/t, cval/t, t, X, Y

if __name__ == '__main__':

    COEF = 10
    EP = 0.1
    SIGMA = 0.001
    SIZE = 10

    g = Grid(SIZE, COEF)
    rval, cval, t, X, Y = DO_Nash(g, EP, SIGMA, "uniform")
    print(g.actions)
    print(g.costs)
    print(rval, cval, t, len(X), len(Y))
    print([g.strategies2[i] for i in range(len(X)) if X[i] > 0])    
    # print([g.strategies1[i] for i in range(len(Y)) if Y[i] > 0])

    g = Grid(SIZE, COEF)
    rval, cval, t, X, Y = DO_Nash(g, EP, 0, "none")
    # print(g.M)
    print(rval, cval, t, len(X), len(Y))
    print([g.strategies2[i] for i in range(len(X)) if X[i] > 0])    

