#!/usr/bin/env python3

import numpy as np

# predict
# x = x
# P = P+Q
#       ^^ q encodes the decay of knowledge over time!

# update
# y = 0
# S = HPH^T + R           // H == [1] which persons did you observe? [2] how correlated are the persons? NxN.
# K = PH^T * S^{-1}       //
# P' = (I-KH)P            // update this guy
# ^^^  we're interested in entropy(P) - entropy(P')


def H_from_G_and_C(G, C, v1=0.1, v2=0.3):
    """
    v1 = measurement variance with direct conversation.
    v2 = measurement variance with indirect conversation.
    #    ^ (1) v2 should be greater than v1
    """
    assert(v2 >= v1)
    # G = group assignment of shape (N,K)
    # C = correlations due to project teams.
    # 1 if on same team, 0 if not on same team.

    # let's think about C later.
    # H = G@G.T# - np.eye(G.shape[0])

    # First-hand
    H1 = G @ G.T

    # Second-hand effects
    H2 = G @ G.T @ C.T @ C - H1

    # H2 = H1 @ (C.T@C - I)
    # H1 * (v1-v2)

    # Covariances of the "measurements"
    H = H1 * v1 + H2 * v2
    return H

# INPUTS:
# P0 : NxN initial covariance.
# A  : attendance, boolean vector of shape [N] indicating whether the person is present.
# H  : group covariance (= how much can you know about the team by talking to the team member?)
# T  : rollout horizon.
# K  : number of groups to create.

# OUTPUTS:
# G  : Group Assignment of shape (T,N,K)

# gaussian update rule = var0*var1 / (var0+var1) right?
# so we should probably assume this is how the uncertainty gets reduced...
# gaussian entropy = 0.5*log(2*pi*sigma^2)

# constraints:
# 1. sum(G, axis=1) == A
# 2. sum(G, axis=0) <= ceil(sum(A)//K)


def transition(P, G, C):
    # observation model
    H = H_from_G_and_C(G, C)

    # Add uncertainty from one timestep.
    # Only for those who are not part of your team.
    # ^ is this fair? idk.
    q_team: float = 0.1
    q_else: float = 0.5
    Q = np.where(C.T @ C, q_team, q_else)
    # Q = 0.001 * (1 - C.T@C)

    P = P + Q
    # Apply gaussian update where observed
    P = np.where(H, (P * H) / (P + H), P)

    # S = H@P@H.T+R
    # K = P@H.T@np.linalg.inv(S)
    # P = (np.eye(*K.shape)-K@H)@P
    return P


def assign_groups(P, A, H, T, K):
    N: int = 15

    persons = []
    teams = []

    # [1] Everyone is 100% certain about themselves.
    P0 = 1.0 - np.eye(N)
    print(P0)

# ... Group assignments ...
# groups = partition N into three components.
# in other words, assign [100], [010], [001] to each member.
# constraints:
# >the sum of each row should be exactly one (if the participant has attended.)
# >the sum of each column(teamn) should be less than ceil(N/3).

# ... We make markov assumption and ignore the history.
# Instead, we assume that C0 adequately captures.


def _entropy(P):
    I = np.eye(6)
    return np.sum(np.where(I, 0, 0.5 * np.log(2 * np.pi * P)))


def evaluate(Gs, P, C):
    """evaluate group assignments."""
    IG = 0.0
    H = _entropy(P)
    for G in Gs:
        P2 = transition(P, G, C)
        H2 = _entropy(P2)# < should this be discounted?
        IG += H - H2

        P = P2
        H = H2
    return IG


def main():
    # [GROUPS]
    G = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0]]
    G = np.asarray(G, dtype=np.float32)

    # [TEAMS]
    C = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]]
    C = np.asarray(C, dtype=np.float32)

    # [COVS]
    P = 1.0 - np.eye(6)
    P2 = transition(P, G, C)
    print(P)
    print(P2)

    I = np.eye(6)
    H = np.sum(np.where(I, 0, 0.5 * np.log(2 * np.pi * P)))
    H2 = np.sum(np.where(I, 0, 0.5 * np.log(2 * np.pi * P2)))

    print(H)
    print(H2)
    print(evaluate(G[None], P, C))


if __name__ == '__main__':
    main()
