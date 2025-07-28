# perturbing-best-responses

# Files

- astar.py implementation of A* algorithm for the path planning game
- examples.py examples of normal-form games and clustered normal-form games for efficient perturbations
- game.py path-planning game on the grid together with SDO and SFP implementations
- perturb_FP.py SFP and SDO implementation for normal-form games
- pqueue.py priority queue for A* algorithm
- zero_sum.py LP solver for zero-sum normal-form games

# Experiments

- SDO_3_2_eff_exp.py SDO with efficient perturbations applied to the example from Theorem 3.2 (Expoential lower bound)
- SDO_Morra_exp.py SDO applied to the Morra game
- SDO_Morra_crisp_exp.py as above but crisp version of the Morra game (maybe it is not a good example)
- SDO_path_planning_exp.py SDO applied to the path-planning game
- SDO_random_exp.py SDO applied to random normal-form games
- SFP_Morra_exp.py SFP applied to the Morra game
- SFP_random_exp.py SFP applied to random normal-form games

# TODO

- extend experiments to investigate influence of the variance
- draw tikz pictures of Markov games from Theorems 3.2 and 3.3
- visualize the clustering of the matrices for the above examples
- visualize path-planning game on the grid, particularly cost of edges and perhaps equilibrium strategies for a small instance