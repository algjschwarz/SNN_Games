import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time

creature_pos = np.random.randint(0, 20, size=2)
food_pos = np.random.randint(0, 20, size=2)
print(creature_pos, food_pos)

fig, ax = plt.subplots(1, 1)
plt.ion()

def update_plot():
    ax.cla()
    ax.axis((0, 20, 0, 20))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.plot(creature_pos[0], creature_pos[1], 'bs')
    ax.plot(food_pos[0], food_pos[1], 'go')
    plt.pause(1)

while not np.array_equal(creature_pos, food_pos):
    if creature_pos[0] < food_pos[0]:
        creature_pos[0] += 1
    elif creature_pos[0] > food_pos[0]:
        creature_pos[0] -= 1
    if creature_pos[1] < food_pos[1]:
        creature_pos[1] += 1
    elif creature_pos[1] > food_pos[1]:
        creature_pos[1] -= 1
    
    update_plot()
