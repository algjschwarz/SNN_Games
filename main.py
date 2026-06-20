import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random

creature_pos = np.random.randint(0, 20, size=2)
food_pos = np.random.randint(0, 20, size=2)
home_pos = np.random.randint(0, 20, size=2)
world_size = (0, 20, 0, 20)

fig, ax = plt.subplots(1, 1)
plt.ion()

class Neuron():
    def __init__(self):
        self.activation_threshold = random.randint(5,10)
        self.voltage = 0.0
        self.weight = random.uniform(2, 10)
        self.bias = random.uniform(1, 5)
        self.leak = 1 - random.uniform(.1, .3)
    def add_voltage(self, voltage):
        self.voltage += voltage
    def attempt_fire(self):
        if self.voltage >= self.activation_threshold:
            self.voltage = 0
            return True
        self.voltage *= self.leak
        return False
    

def update_plot():
    ax.cla()
    ax.axis(world_size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.plot(creature_pos[0], creature_pos[1], 'b^')
    ax.plot(food_pos[0], food_pos[1], 'ro')
    ax.plot(home_pos[0], home_pos[1], 'mP')
    plt.pause(1)

class Creature():
    def __init__(self):
        self.x_pos = Neuron()
        self.x_neg = Neuron()
        self.y_pos = Neuron()
        self.y_neg = Neuron()

        self.neurons = [self.x_pos, self.x_neg, self.y_pos, self.y_neg]

    def smell_food(self):
        spike_array = np.zeros(4)
        x_distance = np.clip(abs(food_pos[0] - creature_pos[0]), 1, 6)
        y_distance = np.clip(abs(food_pos[1] - creature_pos[1]), 1, 6)
        if creature_pos[0] < food_pos[0]:
            spike_array[0] = x_distance
        elif creature_pos[0] > food_pos[0]:
            spike_array[1] = x_distance
        if creature_pos[1] < food_pos[1]:
            spike_array[2] = y_distance
        elif creature_pos[1] > food_pos[1]:
            spike_array[3] = y_distance
        return spike_array
    
    def sense_home(self):
        min_distance = 2
        max_distance = 3

        spike_array = np.zeros(4)
        x_distance = np.clip(abs(home_pos[0] - creature_pos[0]), 1, 6)
        y_distance = np.clip(abs(home_pos[1] - creature_pos[1]), 1, 6)
        distance = np.sqrt(x_distance**2 + y_distance**2)

        if distance in range(min_distance, max_distance):
            return spike_array
        

        return spike_array

    def move_towards_food(self, spike_array):
        for i in range(len(spike_array)):
            if spike_array[i] != 0:
                self.neurons[i].add_voltage(5)
                spike = self.neurons[i].attempt_fire()
                if spike:
                    match i:
                        case 0:
                            creature_pos[0] += spike_array[i]
                        case 1:
                            creature_pos[0] -= spike_array[i]
                        case 2:
                            creature_pos[1] += spike_array[i]
                        case 3:
                            creature_pos[1] -= spike_array[i]


def main():
    creature = Creature()

    while not np.array_equal(creature_pos, food_pos):
        spike_array = creature.smell_food()
        creature.move_towards_food(spike_array)

        update_plot()

main()
