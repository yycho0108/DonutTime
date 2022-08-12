#!/usr/bin/env python3

from datetime import date, datetime

from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, OptSolver
from functools import partial
import itertools
import json
import multiprocessing as mp
from collections import namedtuple
from matplotlib import pyplot as plt
import logging


def _entropy(P):
    out = 0.0
    for key in P:
        if key[0] == key[1]:
            continue
        out += 0.5 * np.log(2 * np.pi * P[key])
    return out
    # I = np.eye(*P.shape)
    # return np.sum(np.where(I, 0, 0.5 * np.log(2 * np.pi * P)))


def _entropy2(P, N: int):
    out = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            out += 0.5 * pyo.log(2 * np.pi * P[i, j])
    return out


def transition_model(P, G, C, Q,
                     r_talk: float, r_hear: float
                     ):
    # Construct H matrix representing
    # variance of independent gaussians
    H1 = G @ G.T
    H2 = G @ G.T @ C.T @ C - H1  # TODO(ycho): check if this is correct
    H = H1 * r_talk + H2 * r_hear

    # Add uncertainty from one timestep.
    # Only for those who are not part of your team.
    P = P + Q

    # Apply gaussian updates where "observed".
    # P = np.where(H, (P * H) / (P + H), P)
    P = ((P * H) + P**2 * (1 - H)) / (P + H)
    return P


def minimize_overlap(model, *,
                     T: int, N: int, K: int, D: int,
                     r_talk: float,
                     r_hear: float,
                     gamma: float = 1.0,
                     rollout: Optional[int] = None,
                     G_prvs: Optional[List[np.ndarray]] = None
                     ):
    """Just count duplicates."""
    if rollout is None:
        rollout = T

    G = np.asarray([[[model.G[i, j, k] for k in range(K)] for j in range(N)]
                    for i in range(T)])
    if G_prvs is not None:
        G = np.concatenate([G_prvs, G], axis=0)
        # NOTE(ycho): rollout only accounts for the future,
        # but now we need to incorporate the history as well. :)
        rollout += len(G_prvs)

    C = np.asarray([
        [model.C[d, n] for n in range(N)]
        for d in range(D)])
    # FIXME(ycho): !!!!HARDCODED WEIGHTING
    # FOR SUBTEAM GROUPS!!!!
    H = [4 * Gi @ Gi.T + C.T @ C for Gi in G]
    # H = [Gi @ Gi.T for Gi in G]
    dup = []

    for dt in range(1, rollout):
        # Linearly weighted (needed?)
        # weight = (T - dt)

        # Exponentially weighted
        # weight = 0.5 ** dt

        # Exponentially weighted,
        # but with integers.
        #, in case it helps :P
        weight = int(4.0 ** (T - dt - 1))
        # assert(weight > 0)
        # print(F'dt={dt}, weight={weight}')
        for i in range(dt, T):
            # dup.append(weight * np.sum(H[i] * H[i - dt]))
            dup.append(weight * sum(sum(H[i] * H[i - dt])))
    return sum(dup)


def minimize_entropy(model, *, T: int, N: int, K: int, D: int,
                     r_talk: float,
                     r_hear: float,
                     gamma: float = 1.0):
    """Minimize entropy after K horizons."""
    # These are parameters
    P = model.P
    C = model.C
    H = _entropy(P)

    # Convert to np.array for more easily defining
    # mathematical operations.
    P = np.asarray([[model.P[i, j] for j in range(N)] for i in range(N)])
    G = np.asarray([[[model.G[i, j, k] for k in range(K)] for j in range(N)]
                    for i in range(T)])
    C = np.asarray([[model.C[j, k] for k in range(N)] for j in range(D)])
    Q = np.asarray([[model.Q[i, j] for j in range(N)] for i in range(N)])

    if gamma < 1.0:
        rewards = []
        for i in range(T):
            P = transition_model(P, G[i], C, Q, r_talk, r_hear)
            H2 = _entropy2(P, N)  # < should this be discounted?
            IG = H - H2
            reward = (gamma ** i) * IG
            H = H2
            rewards.append(reward)
        score = sum(rewards)
    else:
        H0 = H
        for i in range(T):
            P = transition_model(P, G[i], C, Q, r_talk, r_hear)
            H = _entropy2(P, N)  # < should this be discounted?
        score = H0 - H
    return H


def solve_donuts_monte_carlo(
        P: np.ndarray,  # initial covariance
        C: np.ndarray,  # correlation matrix for intra-lab teams
        A: np.ndarray,  # binary variable indicating attendance
        T: int = 1,  # rollout horizon I guess...
        K: int = 3,  # number of groups to create.
        num_iter: int = 16384,
        G_prvs: Optional[List[np.ndarray]] = None,
):
    Model = namedtuple('Model', ('G', 'C'))

    best_cost: float = np.inf
    best_G = None
    costs = []

    N: int = P.shape[0]
    D: int = C.shape[0]
    G = np.zeros((T, N, K), dtype=bool)
    for t in range(T):
        G[t][np.arange(N), np.arange(N) % K] = 1

    for i in range(num_iter):
        # Try random group assignments.
        G[0] = 0
        G[0][A.astype(bool), np.random.permutation(A.sum()) %
             K] = 1  # < attendance based
        for t in range(1, T):
            G[t] = 0
            G[t, np.arange(N), np.random.permutation(N) % K] = 1
        #G[1:] = np.random.permutation(
        #    G[1:].transpose(2, 0, 1)).transpose(1, 2, 0)

        # np.random.permutation(
        # G[1:][:, [np.random.permutation(N) % K for _ in range(T - 1)]] = 1
        # print(i, G)
        model = Model(G=G, C=C)

        # Evaluate cost and track best cost.
        cost = minimize_overlap(model, T=T, N=N, K=K, D=D, r_talk=0, r_hear=0,
                                G_prvs=G_prvs)
        costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_G = G.copy()
    # print(best_G)
    return (best_cost, best_G, costs)


def solve_donuts(
        P: np.ndarray,  # initial covariance
        C: np.ndarray,  # correlation matrix for intra-lab teams
        A: np.ndarray,  # binary variable indicating attendance
        T: int = 1,  # rollout horizon I guess...
        K: int = 3,  # number of groups to create.

        # parameters
        r_talk: float = 0.1,
        r_hear: float = 0.1 + 0.2,
        q_team: float = 0.1,
        q_else: float = 0.5,
        gamma: float = 0.99,
        G0: int = None,
        G_prvs: Optional[List[np.ndarray]] = None
):
    N: int = P.shape[0]
    D: int = C.shape[0]
    assert(P.shape[0] == P.shape[1])

    print('num people', N)
    print('num teams', K)

    Q = np.where(C.T @ C, q_team, q_else)

    # Instantiate model.
    model = pyo.ConcreteModel()

    # Ranges
    model.T = pyo.RangeSet(0, T - 1)  # num horizon
    model.N = pyo.RangeSet(0, N - 1)  # num people
    model.K = pyo.RangeSet(0, K - 1)  # num groups
    model.D = pyo.RangeSet(0, D - 1)  # num sub-teams

    # Parameters
    model.P = pyo.Param(model.N, model.N, initialize=(
        lambda model, i, j: P[i, j]))  # initial covariance
    model.C = pyo.Param(model.D, model.N, initialize=(
        lambda model, i, j: C[i, j]), within=pyo.Binary)  # team assignments
    model.Q = pyo.Param(model.N, model.N, initialize=(
        lambda model, i, j: Q[i, j]))  # team assignments
    model.A = pyo.Param(model.N, initialize=(
        lambda model, i: A[i]), within=pyo.Binary)  # attendance this week

    # Variable for Group assignments
    if G0 is not None:
        print('init')
        model.G = pyo.Var(model.T, model.N, model.K,
                          within=pyo.Binary,
                          initialize=lambda model, i, j, k: G0[i][j, k])
    else:
        model.G = pyo.Var(model.T, model.N, model.K,
                          within=pyo.Binary)

    #obj = partial(minimize_entropy, T=T, N=N, K=K,D=D,
    #              r_talk=r_talk, r_hear=r_hear, gamma=gamma)
    obj = partial(minimize_overlap, T=T, N=N, K=K, D=D,
                  r_talk=r_talk, r_hear=r_hear, gamma=gamma,
                  G_prvs=G_prvs)
    model.objective = pyo.Objective(expr=obj,
                                    # sense=pyo.maximize,
                                    sense=pyo.minimize)
    model.objective.display()

    def _rule_one_group(model, t, i):
        if t == 0:
            # [This week]
            # Everyone who has attended must be assigned to one group.
            return sum(model.G[t, i, j] for j in model.K) == int(A[i])
        # In future weeks, we assume everyone attended.
        # each person may be assigned to only one group.
        return sum(model.G[t, i, j] for j in model.K) == 1
    model.one_group = pyo.Constraint(model.T, model.N,
                                     rule=_rule_one_group)

    # The size of each group should be roughly equal.
    # This is achieved by enforcing |g_i| <= |G|//K.
    def _rule_group_size(model, t, j):
        if t == 0:
            Nt = A.sum()
        else:
            Nt = N
        # TODO(ycho): which is a better objective?
        max_group_size = (Nt + K - 1) // K
        return sum(model.G[t, i, j] for i in model.N) <= max_group_size
    model.group_size_max = pyo.Constraint(model.T, model.K,
                                          rule=_rule_group_size)

    def _rule_group_size_min(model, t, j):
        if t == 0:
            Nt = A.sum()
        else:
            Nt = N
        # TODO(ycho): which is a better objective?
        min_group_size = Nt // K
        return sum(model.G[t, i, j] for i in model.N) >= min_group_size
    model.group_size_min = pyo.Constraint(model.T, model.K,
                                          rule=_rule_group_size_min)

    # Solve.
    solver = pyo.SolverFactory('mindtpy')
    # solver = pyo.SolverFactory('bonmin')
    # solver = pyo.SolverFactory('gurobi', solver_io='python')
    results = solver.solve(model, tee=True,
                           iteration_limit=128,
                           stalling_limit=64,
                           # strategy='OA',
                           # strategy='FP',
                           # init_strategy='FP',
                           # solution_pool=True,
                           # iteration_limit=128,
                           # num_solution_iteration=10,
                           # mip_solver='cplex_persistent', #??
                           # time_limit=180
                           )

    def _value(i, j, k):
        try:
            return pyo.value(model.G[i, j, k])
        except BaseException:
            return 0
    G = np.asarray(
        [[[_value(i, j, k) for k in range(K)] for j in range(N)]
         for i in range(T)])
    if False:
        print('G', G)
        print(P)
        for Gi in G:
            P = transition_model(P, Gi, C, Q, r_talk, r_hear)
            print(P)
    # print('Objective value:')
    # print(pyo.value(model.objective))
    return pyo.value(model.objective), G


def main_test():
    # Fix random seed for experimentation...
    np.random.seed(0)

    # Define Groups.
    G = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0]]
    G = np.asarray(G, dtype=np.float32)

    # Define Teams (project sub-teams)
    C = [[1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1]]
    C = np.asarray(C, dtype=np.float32)

    # Define initial covariance.
    P = 1.0 - np.eye(6)

    # Define today's attendance.
    A = np.random.choice(2, size=6).astype(int)

    solve_donuts(P, C, A, T=3,
                 gamma=1.0)


def solve_rand(seed: int, N: int, T: int, K: int, D: int):
    rng = np.random.default_rng(seed)
    P = 1.0 - np.eye(N)

    # Generate subteams.
    C = np.zeros((D, N), dtype=bool)
    # I = np.random.permutation(N) % D
    I = rng.choice(D, size=N)
    C[I, np.arange(N)] = 1

    # Generate attendance.
    A = rng.choice(2, size=N).astype(bool)
    solve_donuts(P, C, A, T=T, K=K)


def main_rand():
    """solve randomized problem."""
    np.random.seed(0)

    N: int = 17  # num people.
    T: int = 3  # lookahead horizon.
    K: int = 3  # num groups.
    D: int = 5  # number of intra-lab teams.
    solve = partial(solve_rand, N=N, T=T, K=K, D=D)
    print(mp.Pool(8).map_async(solve, range(128)).get())


def main():
    np.random.seed(0)
    # Lookback horizon.
    B: int = 2
    # Lookahead horizon.
    # NOTE(ycho): for whatever reason,
    # when T<=2, the solubility depends on the
    # attendance configuration. This needs to be debugged.
    T: int = 3

    rooms: List[str] = [
        'current room',
        'imcube',
        'elsewhere in the universe',
        # 'one more room!',
        # 'alpha',
        # 'beta'
    ]

    # Number of donut-time discussion
    # groups to create.
    K: int = len(rooms)

    # List of participants..
    persons: List[str] = [
        'beomjoon',
        'changjae',
        'dongchan',  # ?
        'dongryung',
        'dongwon',
        'junhyek',
        'heesang',
        'jaehyung',
        'jamie',
        'jisu',
        'jiyong',
        'junyeob',  # ?
        'minchan',
        'quang-minh',
        'sanghyeon',
        'wonjae',  # ?
        'yoonwoo'
    ]
    N: int = len(persons)

    # Map name to index.
    p2i = {p: i for i, p in enumerate(persons)}

    # Sub-teams.
    # NOTE(ycho): sub-teams are only used for
    # `minimize_entropy` objective,
    # which is disabled because it's too expensive.
    sub_teams: List[List[str]] = [
        # CRM team
        ['minchan', 'junhyek', 'jaehyung'],
        # Packing team
        ['jamie', 'yoonwoo'],
        # POMDP team
        ['jiyong', 'sanghyeon'],
        # IS team
        ['jisu', 'dongryung'],
        # Hierarchical planning team
        ['heesang', 'quang-minh', 'changjae'],
    ]
    D: int = len(sub_teams)
    C = np.zeros((D, N), bool)
    for i in range(D):
        for person in sub_teams[i]:
            C[i, p2i[person]] = 1

    # TODO(ycho):
    # Fill in last two week's groups.

    # Fill in this week's attendance.
    attendance: List[str] = [
        'beomjoon',
        # 'changjae',
        # 'dongchan',
        'dongryung',
        'dongwon',
        'junhyek',
        'heesang',
        'jaehyung',
        'jamie',
        'jisu',
        'jiyong',
        # 'junyeob',
        'minchan',
        'quang-minh',
        'sanghyeon',
        'wonjae',
        # 'yoonwoo'
    ]
    A = np.zeros((N,), dtype=bool)
    for person in attendance:
        A[p2i[person]] = 1

    # Arbitrary P for now..
    # NOTE(ycho): unused for `minimize_overlap` objective.
    P = 1 - np.eye(N)

    # Prepend last weeks' results.
    if B > 0:
        prev_assignments = []
        for f in Path('.').glob('*.json'):
            try:
                prev_date = datetime.strptime(f.stem, '%Y-%m-%d')
                if prev_date.day >= date.today().day:
                    continue
                prev_assignments.append((T, f))
            except Exception as e:
                logging.debug(F'{e}')
                continue
        prev_assignments = sorted(prev_assignments)[-B:]

        G_prvs = []
        for _, filename in prev_assignments:
            with open(filename, 'r') as fp:
                # G = np.zeros((T, N, K), dtype=bool)
                prv: Dict[str, List[str]] = json.load(fp)
                G_prv = []
                for _, v in prv.items():
                    room = np.zeros((N,), dtype=bool)
                    for p in v:
                        room[p2i[p]] = 1
                    G_prv.append(room)
                G_prv = np.stack(G_prv, axis=0).T
                G_prvs.append(G_prv)
        G_prvs = np.stack(G_prvs, axis=0)
    else:
        G_prvs = None

    # Compute group assignments.
    random_cost, GsMC, costs = solve_donuts_monte_carlo(P, C, A, T, K,
                                                        G_prvs=G_prvs)
    best_cost, GsOpt = solve_donuts(P, C, A, T, K,
                                    G0=GsMC,
                                    G_prvs=G_prvs)
    if best_cost < random_cost:
        Gs = GsOpt
        print(
            F'Optimization: Better than best from monte carlo simulation:'
            + F'{best_cost} < {random_cost}'
        )
    else:
        Gs = GsMC
        print(
            F'Optimization: worse than best from monte carlo simulation:'
            + F'{best_cost} >= {random_cost}'
        )
    #plt.hist(costs, bins=32)
    #plt.axvline(x=best_cost, color='r')
    #plt.show()

    # Account for `G_prvs` in below validation...
    if G_prvs is not None:
        Gs = np.concatenate([G_prvs, Gs], axis=0)

    # As validation, look for overlaps.
    mask = 1 - np.eye(N)
    mask[np.triu_indices_from(mask)] = 0
    for G0, G1 in zip(Gs[:-1], Gs[1:]):
        pairs = np.argwhere(np.logical_and.reduce(
            [G0 @ G0.T, G1 @ G1.T, mask]))
        print('>')
        for i, j in pairs:
            print(persons[i], persons[j])

    for t, G in enumerate(Gs):
        if G_prvs is not None:
            t = t - len(G_prvs)
        index = np.argwhere(G)

        groups = [[] for _ in rooms]
        for pi, gi in index:
            groups[gi].append(persons[pi])

        print(F'==week {t:02d} == ')
        for r, g in zip(rooms, groups):
            print(F'\t== room [{r}] == ')
            print(F'\t{g}')

        if t != 0:
            continue

        # Export this week's output.
        today = date.today()
        filename = today.strftime('%Y-%m-%d.json')
        if Path(filename).exists():
            write_output = False
            response = input(
                F'File {filename} already exists! overwrite? (y/N)')
            if (isinstance(response, str) and len(response)
                    > 0 and response[0].lower() == 'y'):
                write_output = True
        if write_output:
            print('writing.')
            with open(filename, 'w') as fp:
                json.dump({r: g for r, g in zip(rooms, groups)}, fp,
                          indent=4)


if __name__ == '__main__':
    # main_rand()
    main()
