# import heapdict as hd
import pqueue as pq

def getPath(state, parent, action, cost):
    path = [state]
    plan = []
    c = 0
    s = state
    while s in parent:
        plan.append(action[s])
        c += cost(action[s])
        s = parent[s]
        path.append(s)
    path.reverse()
    plan.reverse()
    return plan

def astar(s0, isGoal, getApplicable, succ, cost, heur):
    gScore = {}
    parent = {}
    action = {}
    open = pq.PQueue() #hd.heapdict()
    heur_memo = {}

    gScore[s0] = 0
    h = heur(s0)
    if h == None:
        return None
    heur_memo[s0] = h
    open[s0] = h
    m = h
    # print("New heuristic was found %d." % h)
    while open.items():
        # print(list(open.items()))
        (s, fScore) = open.popitem()
        if isGoal(s):
            return getPath(s, parent, action, cost)
        for a in getApplicable(s):
            v = gScore[s] + cost(a)
            t = succ(a, s)
            if not (t in gScore) or v < gScore[t]:
                gScore[t] = v
                parent[t] = s
                action[t] = a
                if t in heur_memo:
                    h = heur_memo[t]
                else:
                    h = heur(t)
                    if h == None:
                        continue
                    heur_memo[t] = h
                open[t] = v + h
                # if h<m:
                #     m = h
                #     print("New heuristic was found %d." % h)

    return None

if __name__ == '__main__':

    edges = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd')]
    cs = [1, 3, 1, 2]
    hs = { 'a': 3, 'b': 3, 'c': 0, 'd': 0 }

    def cost(a):
        for i in range(len(edges)):
            if a == edges[i]:
                return cs[i]
            
    def isGoal(s):
        return s == 'd'

    def getApplicable(s):
        return [ e for e in edges if e[0] == s ]

    def succ(s, a):
        return s[1]

    def heur(s):
        return hs[s]

    plan = astar('a', isGoal, getApplicable, succ, cost, heur)
    print(plan)