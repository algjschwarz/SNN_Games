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
    def __init__(self, activation_threshold, leak):
        self.activation_threshold = activation_threshold
        self.voltage = 0.0
        self.leak = leak
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

class Genome():
    def __init__(self):
        self.neurons = {}
        self.synapses = {}
        self.next_neuron_id = 0

    def add_neuron(self, neuron_type, activation_threshold, leak):
        neuron_id = self.next_neuron_id
        self.next_neuron_id += 1
        self.neurons[neuron_id] = {"type": neuron_type, "activation_threshold": activation_threshold, "leak": leak}

    def add_synapse(self, in_id, out_id, weight):
        self.synapses[(in_id, out_id)] = {"weight": weight}

class Brain():
    def __init__(self, genome):
        self.genome = genome
        self.hidden_neurons = []
        self.sensory_neurons = []
        self.motor_neurons = []
        self.synapses = []
        self.outgoing_synapses = {}
        for i in range(len(self.genome.neurons)):
            activation_threshold = self.genome.neurons[i]["activation_threshold"]
            leak = self.genome.neurons[i]["leak"]
            if genome.neurons[i]["type"] == "hidden":
                self.hidden_neurons.append(Neuron(activation_threshold, leak))
            elif genome.neurons[i]["type"] == "sensory":
                self.sensory_neurons.append(Neuron(activation_threshold, leak))
            elif genome.neurons[i]["type"] == "motor":
                self.motor_neurons.append(Neuron(activation_threshold, leak))
        for (in_id, out_id), weight in self.genome.synapses:
            synapse = Synapse(self.synapse(in_id, out_id, weight))
            self.synapses.append(synapse)
            if synapse not in self.outgoing_synapses:
                self.outgoing_synapses[synapse.source] = []
            self.outgoing_synapses[synapse.source].append(synapse)

    def propagate(self, neuron, queue):
        for synapse in self.outgoing_synapses.get(neuron, []):
            queue.append(synapse.target)

    def step(self, sensory_spike_array):
        queued_neurons = []
        output_spike_array = np.zeros(len(self.motor_neurons))
        spike_count = 0

        for i in range(len(self.sensory_neurons)):
            self.sensory_neurons[i].add_voltage(sensory_spike_array[i])
            if self.sensory_neurons[i].attempt_fire():
                self.propagate(self.sensory_neurons[i], queued_neurons)
                spike_count += 1

        while len(queued_neurons) != 0:
            neuron = queued_neurons.pop(0)
            if neuron.attempt_fire():
                if neuron.type == "motor":
                    output_spike_array[self.motor_neurons.index(neuron)] = 1
                else:
                    self.propagate(neuron, queued_neurons)
                spike_count += 1

        return output_spike_array, spike_count

class Creature():
    def __init__(self, genome):
        self.brain = Brain(genome)
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

def generate_genome_no_connections(num_input_neurons, num_output_neurons):
    total_neuron_count = num_input_neurons + num_output_neurons
    genome = Genome()
    activation_thresholds = [random.randint(2, 8) for _ in range(total_neuron_count)]
    leaks = [1 - random.uniform(.1, .3) for _ in range(total_neuron_count)]
    for i in range(num_input_neurons):
        genome.add_neuron("sensory", activation_thresholds[0], leaks[0])
        activation_thresholds.pop(0)
        leaks.pop(0)
    for i in range(num_output_neurons):
        genome.add_neuron("motor", activation_thresholds[i], leaks[i])
    return genome

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
    genome = generate_genome_no_connections(4, 4)
    objects = []
    creatures = [Creature(genome)]
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
        
        update_plot(objects)
main()
