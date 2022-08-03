#!/usr/bin/env python3

from datetime import date

from typing import List
import numpy as np
import pyomo.environ as pyo
from functools import partial
import itertools
import json


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


def minimize_overlap(model, *, T: int, N: int, K: int,
                     r_talk: float,
                     r_hear: float,
                     gamma: float = 1.0):
    """Just count duplicates."""
    G = np.asarray([[[model.G[i, j, k] for k in range(K)] for j in range(N)]
                    for i in range(T)])
    # H = np.einsum('bij,bkj->bik', G, G)
    H = [Gi @ Gi.T for Gi in G]
    dup = []

    for dt in range(1, T):
        # Linearly weighted (needed?)
        # weight = (T - dt)

        # Exponentially weighted
        # weight = 0.5 ** dt

        # Exponentially weighted,
        # but with integers.
        #, in case it helps :P
        weight = 2.0 ** (T - dt - 1)
        for i in range(dt, T):
            dup.append(weight * np.sum(H[i] * H[i - dt]))
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
        ss = []
        for i in range(T):
            P = transition_model(P, G[i], C, Q, r_talk, r_hear)
            H2 = _entropy2(P, N)  # < should this be discounted?
            IG = H - H2
            reward = (gamma ** i) * IG
            H = H2
            ss.append(reward)
        score = sum(ss)
    else:
        H0 = H
        for i in range(T):
            P = transition_model(P, G[i], C, Q, r_talk, r_hear)
            H = _entropy2(P, N)  # < should this be discounted?
        score = H0 - H
    return H


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
        gamma: float = 0.99
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
    model.G = pyo.Var(model.T, model.N, model.K,
                      within=pyo.Binary)

    #obj = partial(minimize_entropy, T=T, N=N, K=K,
    #              r_talk=r_talk, r_hear=r_hear, gamma=gamma)
    obj = partial(minimize_overlap, T=T, N=N, K=K,
                  r_talk=r_talk, r_hear=r_hear, gamma=gamma)
    model.obj = pyo.Objective(expr=obj,
                              # sense=pyo.maximize,
                              sense=pyo.minimize)

    #def _rule_attendance(model, i):
    #    #return sum(model.G[0, i, j] for j in model.K) <= model.A[i]
    #    #if model.A[i]:
    #    if A[i]:
    #        return sum(model.G[0, i, j] for j in model.K) == 1
    #    else:
    #        return sum(model.G[0, i, j] for j in model.K) == 0
    #model.attendance = pyo.Constraint(model.N,
    #                                  rule=_rule_attendance)
    def _rule_one_group(model, t, i):
        if t == 0:
            # [This week]
            # Everyone who has attended must be assigned to one group.
            if A[i]:
                return sum(model.G[0, i, j] for j in model.K) == 1
            else:
                return sum(model.G[0, i, j] for j in model.K) == 0
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
        # max_group_size = (Nt + K - 1) // K
        # return sum(model.G[t, i, j] for i in model.N) <= max_group_size
        min_group_size = Nt // K
        return sum(model.G[t, i, j] for i in model.N) >= min_group_size
    model.group_size = pyo.Constraint(model.T, model.K,
                                      rule=_rule_group_size)

    # Solve.
    solver = pyo.SolverFactory('mindtpy')
    results = solver.solve(model, tee=True,
                           time_limit=180)
    print(results)

    def _value(i, j, k):
        try:
            return pyo.value(model.G[i, j, k])
        except BaseException:
            return 0
    G = np.asarray(
        [[[_value(i, j, k) for k in range(K)] for j in range(N)]
         for i in range(T)])
    print('G', G)

    # print(P)
    for Gi in G:
        P = transition_model(P, Gi, C, Q, r_talk, r_hear)
        # print(P)
    # print(pyo.value(model.obj))

    return G[0]


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


def main_rand():
    """solve randomized problem."""
    np.random.seed(0)

    N: int = 15  # num people.
    T: int = 3  # lookahead horizon
    K: int = 3  # num groups.
    D: int = 4  # number of intra-lab teams.
    # packing team
    # CRM team

    P = 1.0 - np.eye(N)
    C = np.zeros((D, N), dtype=bool)
    I = np.random.permutation(N) % D
    C[I, np.arange(N)] = 1
    A = np.random.choice(2, size=N).astype(bool)

    solve_donuts(P, C, A, T=T, K=K)


def main():
    # Lookahead horizon.
    T: int = 4

    rooms: List[str] = [
        'current room',
        'imcube',
        'elsewhere in the universe'
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
        'dongchan',
        'dongryung',
        'dongwon',
        'junhyek',
        'heesang',
        'jaehyung',
        'jamie',
        'jisu',
        'jiyong',
        'junyeob',
        'minchan',
        'quang-minh',
        'sanghyeon',
        'wonjae',
        'yoonwoo'
    ]
    A = np.zeros((N,), dtype=bool)
    for person in attendance:
        A[p2i[person]] = 1

    # Arbitrary P for now..
    P = 1 - np.eye(N)

    # Compute group assignments.
    groups = [[] for _ in rooms]
    G = solve_donuts(P, C, A, T, K)
    index = np.argwhere(G)
    for pi, gi in index:
        groups[gi].append(persons[pi])

    for room, persons in zip(rooms, groups):
        print(F'== room [{room}] == ')
        print(F'[{persons}]')

    # Export output.
    today = date.today()
    filename = today.strftime('%Y-%m-%d.json')
    with open(filename, 'w') as fp:
        json.dump({r: g for r, g in zip(rooms, groups)}, fp,
                  indent=4)


if __name__ == '__main__':
    main()
    # main_rand()
