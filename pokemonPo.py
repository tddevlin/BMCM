
# coding: utf-8

# # Pokemon Po

# In[156]:

import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import csv


# In[14]:

walking_speed = 0.129 # blocks/min


# In[15]:

df = pd.read_csv('Providence_Pokemon_1.csv', names = ["xCoord", "yCoord", "points", "time"])
df = df[:-1] # remove 0 row
df.head()


# In[16]:

first12 = df[df['time'] <= 12 * 60]
first12


# The function `inRange` determines whether you can make it from the starting coordinates, `start`, to the ending coordinates, `end`, in `time` minutes or less. If so, it returns 1 and the amount of time required. If not, it returns (0, 0).

# In[32]:

def inRange(start, end, time, speed):
    xDist = abs(end[0] - start[0])
    yDist = abs(end[1] - start[1])
    elapsed = (xDist + yDist)/speed
    if elapsed <= time:
        return 1, elapsed
    else:
        return 0, 0


# `simpleAlgorithm` takes in the current location, `current`, and the walking speed, `speed`, 

# In[33]:

def simpleAlgorithm(current_location, speed, pokeData):
    pokemon_caught = 0
    points = 0
    time = 0
    for index, row in pokeData.iterrows():
        pokemon_location = (row["xCoord"], row["yCoord"])
        arrival_time = row["time"]
        possible_time = min(arrival_time + 15 - time, 15)
        within_range, travel_time = inRange(current_location, pokemon_location, possible_time, speed)
        if within_range:
            pokemon_caught += 1
            points += row["points"]
            time += travel_time
            current_location = pokemon_location
        else:
            time = arrival_time
    return pokemon_caught, points


# ### Test cases

# In[47]:

testData = [[1, 1, 5 , 0],
            [1, 2, 3 , 1],
            [2, 2, 22, 1.5],
            [2, 1, 10, 2],
            [1, 2, 1 , 3]]
testdf = pd.DataFrame(testData, columns = ["xCoord", "yCoord", "points", "time"])
testdf


# In[41]:

simpleAlgorithm((1,1), 1/15, testdf)


# In[42]:

simpleAlgorithm((3,8), 0.129, df)


# In[43]:

def complicatedAlgorithm(current_location, speed, pokeData):
    pokemon_caught = 0
    points = 0
    time = 0
    maxTime = max(pokeData["time"])
    while time < maxTime:
        pokemon_on_board = pokeData[(pokeData["time"] <= time) & (pokeData["time"] >= time - 15)]
        bestPoints = 0
        bestLocation = None
        for index, row in pokemon_on_board.iterrows():
            pokemon_location = [row["xCoord"], row["yCoord"]]
            if pokemon_location == current_location:
                pokemon_caught += 1
                points += row["points"]
            arrival_time = row["time"]
            possible_time = min(arrival_time + 15 - time, 15)
            within_range, travel_time = inRange(current_location, pokemon_location, possible_time, speed)
            if within_range and row["points"] > bestPoints:
                bestPoints = row["points"]
                bestLocation = pokemon_location
        if within_range:
            target_node = pokemon_location
        else:
            target_node = best_node(current_location[0], current_location[1])
        
        # move toward target node
        if current_location[0] != target_node[0]:
            current_location[0] += np.sign(target_node[0] - current_location[0])
        else:
            current_location[1] += np.sign(target_node[1] - current_location[1])
        time += 1/speed # add time required to cross edge
        
    return pokemon_caught, points


# In[44]:

def best_node(x, y):
    left = score(x - 1, y)
    right = score(x + 1, y)
    down = score(x, y - 1)
    up = score(x, y + 1)
    best = max(left, right, down, up)
    if best == left:
        return (x-1, y)
    elif best == right:
        return (x+1, y)
    elif best == down:
        return (x, y-1)
    elif best == up:
        return (x, y+1)


# In[84]:

node_score = np.zeros((12, 12))
for index, row in df.iterrows():
    node_score[row["xCoord"], row["yCoord"]] += row["points"]


# In[103]:

plt.style.use('ggplot')
fig = plt.gca()
fig.set_aspect('equal')
fig.set_xticks(np.array([1,2,3,4,5,6,7,8,9,10]) + 0.5)
fig.set_xticklabels([1,2,3,4,5,6,7,8,9,10])
fig.set_yticks(np.array([1,2,3,4,5,6,7,8,9,10]) + 0.5)
fig.set_yticklabels([1,2,3,4,5,6,7,8,9,10])
fig.set_title("Total point value over 42 days", size=14)
plt.pcolor(node_score ,cmap=plt.cm.Reds)
plt.colorbar()
#plt.show()
plt.savefig("totalPoint.png")


# In[79]:

node_number = np.zeros((12, 12))
for index, row in df.iterrows():
    node_number[row["xCoord"], row["yCoord"]] += 1


# In[112]:

plt.style.use('ggplot')
fig = plt.gca()
fig.set_aspect('equal')
fig.set_xticks(np.array([1,2,3,4,5,6,7,8,9,10]) + 0.5)
fig.set_xticklabels([1,2,3,4,5,6,7,8,9,10])
fig.set_yticks(np.array([1,2,3,4,5,6,7,8,9,10]) + 0.5)
fig.set_yticklabels([1,2,3,4,5,6,7,8,9,10])
fig.set_title("Total number of Pokemon over 42 days", size=14)
plt.pcolor(node_number ,cmap=plt.cm.Reds)
plt.colorbar()
#plt.show()
plt.savefig("totalNumber.png", padinches=0)


# In[152]:

plt.style.use('ggplot')
plt.hist(df["points"], bins=20, histtype='stepfilled', normed=True, color='black')
plt.title("Empirical distribution of point values", size=18)
plt.xlabel("Point value")
plt.ylabel("Frequency")
plt.savefig("Spmf.png")


# In[148]:

timeBetween = np.zeros(len(df["time"]) - 1)
for i in range(len(df["time"])):
    if i == len(df["time"])-1:
        break
    timeBetween[i] = df["time"][i+1] - df["time"][i]


# In[149]:

len(timeBetween)


# In[168]:

mu = np.mean(timeBetween)
sigma2 = np.var(timeBetween)
y = mlab.normpdf(np.linspace(0,70,1000), mu, np.sqrt(sigma2))


# In[169]:

plt.style.use('ggplot')
plt.hist(timeBetween, bins=20, histtype='stepfilled', normed=True, color='black')
plt.title("Empirical distribution of time intervals", size=16)
plt.xlabel("Time between Pokemon appearances")
plt.ylabel("Frequency")
plt.plot(np.linspace(0,70,1000), y, 'r-', linewidth=2)
plt.savefig("interval_hist.png")


# In[114]:

def score(x, y):
    return node_score[x, y]


# In[48]:

complicatedAlgorithm([3,8], walking_speed, df)


# In[49]:

def globalAlgorithm(current_location, speed, pokeData):
    pokemon_caught = 0
    points = 0
    time = 0
    maxTime = max(pokeData["time"])
    while time < maxTime:
        pokemon_on_board = pokeData[(pokeData["time"] <= time) & (pokeData["time"] >= time - 15)]
        bestPoints = 0
        bestLocation = None
        for index, row in pokemon_on_board.iterrows():
            pokemon_location = [row["xCoord"], row["yCoord"]]
            if pokemon_location == current_location:
                pokemon_caught += 1
                points += row["points"]
            arrival_time = row["time"]
            possible_time = min(arrival_time + 15 - time, 15)
            within_range, travel_time = inRange(current_location, pokemon_location, possible_time, speed)
            if within_range and row["points"] > bestPoints:
                bestPoints = row["points"]
                bestLocation = pokemon_location
        if within_range:
            target_node = pokemon_location
        else:
            target_node = [3,8]
        
        # move toward target node
        if current_location[0] != target_node[0]:
            current_location[0] += np.sign(target_node[0] - current_location[0])
        else:
            current_location[1] += np.sign(target_node[1] - current_location[1])
        time += 1/speed # add time required to cross edge
        
    return pokemon_caught, points


# In[50]:

globalAlgorithm([3,8], walking_speed, df)


# In[ ]:



