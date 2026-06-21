import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random

CREATURE_IMG = mpimg.imread('sprites/mouse.png')
FOOD_IMG     = mpimg.imread('sprites/apple.png')
HOME_IMG     = mpimg.imread('sprites/house.png')

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

class Synapse():
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight

class Brain():
    def __init__(self, num_of_neurons):
        self.neurons = [Neuron() for _ in range(num_of_neurons)]
        self.sensory_neuron_index = [i for i in range(0, 4)]
        self.motor_neuron_index = [i for i in range(4,8)]
        self.synapses = [Synapse(self.neurons[i], self.neurons[i + 4], 7) for i in range(4)]

    def propagate(self, neuron, queue) -> list:
        for synapse in self.synapses:
            if synapse.source == neuron:
                if synapse.target:
                    synapse.target.add_voltage(synapse.weight)
                    queue.append(synapse.target)

    def step(self, sensory_spike_array):
        queued_neurons = []
        output_neuron_index = []
        output_spike_array = np.zeros(len(self.motor_neuron_index))
        spike_count = 0

        for i in self.sensory_neuron_index:
            self.neurons[i].add_voltage(sensory_spike_array[i])
            if self.neurons[i].attempt_fire():
                spike_count += 1
                self.propagate(self.neurons[i], queued_neurons)

        while len(queued_neurons) != 0:
            neuron = queued_neurons.pop(0)
            if neuron.attempt_fire():
                self.propagate(neuron, queued_neurons)
                index = self.neurons.index(neuron)
                if index in self.motor_neuron_index:
                    spike_count += 1
                    output_neuron_index.append(index)

        for i in range(len(output_neuron_index)):
            output_spike_array[self.motor_neuron_index.index(output_neuron_index[i])] = 1

        return output_spike_array, spike_count

class Creature():
    def __init__(self):
        self.brain = Brain(8)
        self.image = CREATURE_IMG
        self.zoom = 0.2
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
    
    def eat_food(self, sensory_spike_array):
        if max(sensory_spike_array) == 0:
            energy = self.energy + 20
            self.energy = max(0, min(energy, 30))
            return True
        return False

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

    def think(self, sensory_spike_array) -> np.array:
        movement_commands, spike_count = self.brain.step(sensory_spike_array)
        self.energy -= spike_count * .1
        return movement_commands

    def move(self, movement_commands):
        moved = False
        for i in range(len(movement_commands)):       
            if movement_commands[i] != 0:
                moved = True
                match i:
                    case 0: self.position[0] += 1
                    case 1: self.position[0] -= 1
                    case 2: self.position[1] += 1
                    case 3: self.position[1] -= 1
        if moved:
            self.energy -= 1

class Food():
    def __init__(self):
        self.position = np.random.randint(0, 20, size=2)
        self.image = FOOD_IMG
        self.zoom = .15

class Home():
    def __init__(self):
        self.position = np.random.randint(0, 20, size=2)
        self.image = HOME_IMG
        self.zoom = 0.2

fig, ax = plt.subplots(1, 1)
plt.ion()
world_size = (0, 20, 0, 20)

def update_plot(objects):
    ax.cla()
    ax.axis(world_size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    for obj in objects:
        ab = AnnotationBbox(OffsetImage(obj.image, zoom=obj.zoom),
                            (obj.position[0], obj.position[1]),
                            frameon=False)
        ax.add_artist(ab)
    plt.pause(.5)

def main():
    objects = []
    creatures = [Creature()]
    foods = [Food() for _ in range(3)]
    homes = [Home()]

    while foods:
        objects = creatures + foods + homes
        sensory_data = creatures[0].smell_food(foods)
        if creatures[0].eat_food(sensory_data):
            foods = [f for f in foods if not np.array_equal(creatures[0].position, f.position)]
        else:
            movement_data = creatures[0].think(sensory_data)
            creatures[0].move(movement_data)
        print(creatures[0].energy)
        
        update_plot(objects)
main()
