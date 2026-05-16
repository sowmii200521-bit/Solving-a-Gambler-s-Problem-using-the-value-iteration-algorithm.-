# Ex-4: Solving Gambler’s Problem using Value Iteration

## Aim:
To implement the Value Iteration Algorithm for solving the Gambler’s Problem in Reinforcement Learning and determine the optimal policy that maximizes the probability of reaching the goal.

---

## Algorithm:

**Step-1:** Initialize the environment:
   - Define goal amount.
   - Define probability of winning.
   - Initialize value function and policy.

**Step-2:** Initialize state values:
   - Set value of goal state to 1.
   - Set all other state values to 0.

**Step-3:** Perform Value Iteration:
   - For each state:
     - Evaluate all possible stakes.
     - Compute expected returns for each action.
     - Update the value function with the maximum expected return.

**Step-4:** Repeat the iteration until the value function converges.

**Step-5:** Derive the optimal policy:
   - Select the action with the highest value for each state.

**Step-6:** Display the optimal value function and policy.

---

## Program:

```python

#Solving a Gambler’s Problem using the value iteration algorithm. 

import numpy as np

GOAL = 100

# Probability of winning
p_h = 0.4

# Discount factor
gamma = 1.0

# State values
V = np.zeros(GOAL + 1)

# Goal state
V[GOAL] = 1

# Policy
policy = np.zeros(GOAL + 1, dtype=int)

theta = 0.0001


# -------------------------------
# VALUE ITERATION
# Bellman Optimality Update
# -------------------------------

while True:

    delta = 0

    for s in range(1, GOAL):

        old_value = V[s]

        action_returns = []

        # Possible actions
        actions = range(1, min(s, GOAL - s) + 1)

        for a in actions:

            win_state = s + a
            lose_state = s - a

            reward = 0

            if win_state == GOAL:
                reward = 1

            # V(s) = max_a [ p_h(r+γV(s+a))
            #               +(1-p_h)(r+γV(s-a)) ]

            value = (
                p_h * (reward + gamma * V[win_state])
                + (1 - p_h) * (reward + gamma * V[lose_state])
            )

            action_returns.append(value)

        V[s] = max(action_returns)

        delta = max(delta, abs(old_value - V[s]))

    if delta < theta:
        break


# -------------------------------
# EXTRACT OPTIMAL POLICY
# -------------------------------

for s in range(1, GOAL):

    actions = range(1, min(s, GOAL - s) + 1)

    best_action = 0
    best_value = -1

    for a in actions:

        win_state = s + a
        lose_state = s - a

        reward = 0

        if win_state == GOAL:
            reward = 1

        val = (
            p_h * (reward + gamma * V[win_state])
            + (1 - p_h) * (reward + gamma * V[lose_state])
        )

        if val > best_value:
            best_value = val
            best_action = a

    policy[s] = best_action


print("Optimal State Values:\n")
print(V)

print("\nOptimal Betting Policy:\n")
print(policy)
        
```

---

## Output:

```python

Optimal State Values:

[0.         0.00722634 0.01807376 0.03228824 0.04518551 0.06084823
 0.08072238 0.0973491  0.11296378 0.13189529 0.1521217  0.17623956
 0.20180705 0.2283358  0.24337294 0.26050894 0.28240946 0.30313717
 0.32974373 0.36100148 0.38030536 0.4058823  0.44060089 0.46752938
 0.50451763 0.56       0.57084426 0.58711131 0.60843343 0.62777827
 0.65127302 0.68108423 0.70602377 0.72944568 0.75784624 0.78818322
 0.82436053 0.86271058 0.90250655 0.92506006 0.95076381 0.98361426
 1.01470774 1.05461632 1.10150393 1.13045829 1.16882465 1.22090236
 1.26129479 1.31677687 1.4        1.41084426 1.42711131 1.44843343
 1.46777827 1.49127302 1.52108423 1.54602377 1.56944568 1.59784624
 1.62818322 1.66436053 1.70271058 1.74250655 1.76506006 1.79076381
 1.82361426 1.85470774 1.89461632 1.94150393 1.97045829 2.00882465
 2.06090236 2.10129479 2.15677687 2.24       2.25626678 2.28066696
 2.31265054 2.34166741 2.37690993 2.42162635 2.45903603 2.49416856
 2.53676979 2.58227497 2.63654142 2.69406612 2.75376007 2.78759032
 2.82614596 2.87542162 2.92206188 2.98192485 3.05225604 3.09568757
 3.15323713 3.23135363 3.29194228 3.37516537 1.        ]

Optimal Betting Policy:

[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25  1 27 28 29 30  6  7  8 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
 22 23 24 25  1  2  3  4  5  6  7  8  9 10 11 12 12 11 10  9  8  7  6  5
  4  3  2  1  0]

```

---

## Result:

The Value Iteration Algorithm was successfully implemented for the Gambler’s Problem. The optimal value function and policy were obtained, maximizing the probability of reaching the goal state through optimal betting decisions.
