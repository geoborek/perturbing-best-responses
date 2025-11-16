# perturbing-best-responses

This repository contains the code accompanying the following paper:

> A. Dziwoki and R. Horcik. Perturbing Best Responses in Zero-Sum Games. AAAI 2026.

# Experiments

- SFP_random_exp.py SFP applied to random normal-form games
- SFP_U_exp.py SFP applied to the matrix L Example 1 and U^T Example 2
- SFP_Morra_exp.py SFP applied to the Morra game
- SDO_random_exp.py SDO applied to random matrix games
- SDO_L_U_eff_exp.py SDO with efficient perturbations applied to the matrix L and U^T
- SDO_Blotto.py SDO applied to the Blotto game 
- SDO_Morra_exp.py SDO applied to the f-finger Morra game
- SDO_Morra_crisp_exp.py as above but crisp version of the f-finger Morra game
- SDO_path_planning_exp.py SDO applied to the path-planning game

# Files

- astar.py implementation of A* algorithm for the path planning game
- examples.py examples of normal-form games and clustered normal-form games for efficient perturbations
- game.py path-planning game on the grid together with SDO and SFP implementations
- perturb_FP.py FP, DO, AFP, SFP, SDO, and SAFP implementation for normal-form games
- pqueue.py priority queue for A* algorithm
- zero_sum.py LP solver for zero-sum normal-form games (uses scipy LP solver)

