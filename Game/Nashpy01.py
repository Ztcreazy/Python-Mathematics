# https://towardsdatascience.com/game-theory-in-python-with-nashpy-cb5dceab262c
# “Game Theory” is an analysis of strategic interaction. It consists of analyzing or 
# modelling strategies and outcomes based on certain rules, in a game of 2 or 
# more players.It has widespread applications and is useful in political scenarios, 
# logical parlour games, business as well as to observe economic behaviour.
# 
#
#
#                       Player B  
#                       Left    Right
# Player A    Top       (2,4)    (0, 2)
#             Bottom    (4,2)    (2, 0)
# For Player A , it is always better to play “Bottom” because his payoffs (4 or 2) 
# from this strategy is always greater than Top (2 or 0). Similarly, for Player B, 
# it is always better to play “Left” because his payoffs (4 or 2) is always greater than 
# (2 or 0). Hence, the equilibrium strategy is for Player A to play “Bottom” and Player B to 
# play “Left”. This brings us to the concept of dominant strategy.
"""
A pair of strategies is said to be Nash equilibrium (NE), if optimal choice of Player A 
given Player B’s choice coincides with optimal choice of Player B given Player A’s choice. 
In simple terms, initially neither player knows what the other player will do when deciding or 
making a choice. Hence NE is a pair of choices/strategies/expectations where 
neither player wants to change their behaviour even after the strategy/choice of 
the other player is revealed.
"""
import numpy as np
import nashpy as nash

# Create the game with the payoff matrix
A = np.array([[2,0],[4,2]]) # A is the row player
B = np.array([[4,2],[2,0]]) # B is the column player
game1 = nash.Game(A,B)
print("game1: ", game1)

# Find the Nash Equilibrium with Support Enumeration
equilibria = game1.support_enumeration()
for eq in equilibria:
    print("equilibria: ", eq)



# Create the payoff matrix
A = np.array([[4,0],[0,2]]) # A is the row player
B = np.array([[2,0],[0,4]]) # B is the column player
game2 = nash.Game(A,B)
print("game2: ", game2)
# Find the Nash Equilibrium with Support Enumeration
equilibria = game2.support_enumeration()
for eq in equilibria:
    print("equilibria: ", eq)