import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
class Neuron():
    def __init__(self):
        self.activation_threshold = 5
        self.voltage = 0.0
        self.leak = 0.95
    def add_voltage(self, voltage):
        self.voltage += voltage
    def attempt_fire(self):
        if self.voltage >= self.activation_threshold:
            self.voltage = 0
            return True
        self.voltage *= self.leak
        return False
    
class Creature():
    def __init__(self):
        self.neurons = [Neuron() for _ in range(4)]
        self.format_string = 'b^'
        self.position = np.random.randint(0, 20, size=2)
        self.energy = 20 

    def smell_food(self, foods):
        sensory_spike_array = np.zeros(4)
        closest_food = min(foods, key= lambda food: np.linalg.norm(self.position - food.position))

        if self.position[0] < closest_food.position[0]: sensory_spike_array[0] = 1
        elif self.position[0] > closest_food.position[0]: sensory_spike_array[1] = 1

        if self.position[1] < closest_food.position[1]: sensory_spike_array[2] = 1
        elif self.position[1] > closest_food.position[1]: sensory_spike_array[3] = 1

        return sensory_spike_array

    def sense_home(self):
        pass
    """ min_distance = 2
        max_distance = 3
        x_distance = home_pos[0] - self.position[0]
        y_distance = home_pos[1] - self.position[1]
        spike_array = np.zeros(4)
        x_distance_capped = np.clip(abs(x_distance), 1, 6)
        y_distance_capped = np.clip(abs(y_distance), 1, 6)
        distance = np.sqrt(x_distance**2 + y_distance**2)
        if distance in range(min_distance, max_distance):
            return spike_array
        if self.position[0] < home_pos[0]:
            spike_array[0] = x_distance_capped
        elif self.position[0] > home_pos[0]:
            spike_array[1] = x_distance_capped

        if self.position[1] < home_pos[1]:
            spike_array[2] = y_distance_capped
        elif self.position[1] > home_pos[1]:
            spike_array[3] = y_distance_capped

        if self.position[0] == home_pos[0] and self.position[1] == home_pos[1]:
            spike_array[1] = min_distance
        return spike_array """

    def think(self, sensory_spike_array):
        movement_spike_array = np.zeros(4)
        for i in range(len(sensory_spike_array)):
            if sensory_spike_array[i] != 0:
                self.neurons[i].add_voltage(sensory_spike_array[i])
                if self.neurons[i].attempt_fire():
                    movement_spike_array[i] = 1

        return movement_spike_array

    def move(self, movement_spike_array):
        for i in range(len(movement_spike_array)):
            if movement_spike_array[i] != 0:
                match i:
                    case 0: self.position[0] += 1
                    case 1: self.position[0] -= 1
                    case 2: self.position[1] += 1
                    case 3: self.position[1] -= 1

class Food():
    def __init__(self):
        self.position = np.random.randint(0, 20, size=2)
        self.format_string = 'ro'

class Home():
    def __init__(self):
        self.position = np.random.randint(0, 20, size=2)
        self.format_string = 'mP'

fig, ax = plt.subplots(1, 1)
plt.ion()
world_size = (0, 20, 0, 20)

def update_plot(objects):
    ax.cla()
    ax.axis(world_size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    for object in objects:
        ax.plot(object.position[0], object.position[1], object.format_string)
    plt.pause(1)

def main():
    objects = []
    creatures = [Creature()]
    foods = [Food() for _ in range(3)]
    homes = [Home()]

    while True:
        objects = creatures + foods + homes
        sensory_spike_array = creatures[0].smell_food(foods)
        movement_spike_array = creatures[0].think(sensory_spike_array)
        creatures[0].move(movement_spike_array)
        
        update_plot(objects)
main()
