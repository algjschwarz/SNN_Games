import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random
import math
import operator
from dataclasses import dataclass

CREATURE_IMG = mpimg.imread('sprites/mouse.png')
FOOD_IMG     = mpimg.imread('sprites/apple.png')
HOME_IMG     = mpimg.imread('sprites/house.png')

class Neuron():
    def __init__(self, activation_threshold, leak, type, transmitter="excitatory"):
        self.activation_threshold = activation_threshold
        self.voltage = 0.0
        self.leak = leak
        self.type = type
        self.transmitter = transmitter
    def add_voltage(self, voltage):
        self.voltage += voltage
    def attempt_fire(self):
        if self.voltage >= self.activation_threshold:
            self.voltage = 0
            return True
        self.voltage *= self.leak
        return False

@dataclass
class NeuronGene:
    type: str
    activation_threshold: float
    leak: float
    transmitter: str = "excitatory"

    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)

@dataclass
class Synapse:
    source: object
    target: object
    weight: float

class Genome():
    def __init__(self):
        self.neurons = {}
        self.synapses = {}
        self.next_neuron_id = 0

    def add_neuron(self, neuron_type, activation_threshold, leak, transmitter="excitatory"):
        neuron_id = self.next_neuron_id
        self.next_neuron_id += 1
        self.neurons[neuron_id] = NeuronGene(neuron_type, activation_threshold, leak, transmitter)

    def add_synapse(self, in_id, out_id, weight):
        self.synapses[(in_id, out_id)] = {"weight": weight}

class Brain():
    def __init__(self, genome):
        self.hidden_neurons = []
        self.sensory_neurons = []
        self.motor_neurons = []
        self.synapses = []
        self.outgoing_synapses = {}

        neuron_by_id = {}
        for i in range(len(genome.neurons)):
            activation_threshold = genome.neurons[i]["activation_threshold"]
            leak = genome.neurons[i]["leak"]
            new_neuron = Neuron(activation_threshold, leak, genome.neurons[i]["type"])
            neuron_by_id[i] = new_neuron
            if genome.neurons[i]["type"] == "hidden":
                self.hidden_neurons.append(new_neuron)
            elif genome.neurons[i]["type"] == "sensory":
                self.sensory_neurons.append(new_neuron)
            elif genome.neurons[i]["type"] == "motor":
                self.motor_neurons.append(new_neuron)
        for synapse_key in genome.synapses.keys():
            weight = genome.synapses[synapse_key]["weight"]
            neuron_a = neuron_by_id[synapse_key[0]]
            neuron_b = neuron_by_id[synapse_key[1]]
            
            synapse = Synapse(neuron_a, neuron_b, weight)
            self.synapses.append(synapse)
            if synapse.source not in self.outgoing_synapses:
                self.outgoing_synapses[synapse.source] = []
            self.outgoing_synapses[synapse.source].append(synapse)

    def propagate(self, neuron, queue):
        for synapse in self.outgoing_synapses.get(neuron, []):
            queue.append([synapse.target, synapse.weight])

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
            next = queued_neurons.pop(0)
            neuron = next[0]
            weight = next[1]
            neuron.add_voltage(weight)
            if neuron.attempt_fire():
                if neuron.type == "motor":
                    output_spike_array[self.motor_neurons.index(neuron)] = 1
                else:
                    self.propagate(neuron, queued_neurons)
                spike_count += 1

        return output_spike_array, spike_count

class Creature():
    def __init__(self, genome, world_size=(0,20,0,20)):
        self.genome = genome
        self.brain = Brain(genome)
        self.image = CREATURE_IMG
        self.zoom = 0.2
        self.position = np.random.randint(world_size[0], world_size[1], size=2)
        self.energy = 20 

    def is_dead(self) -> bool:
        if self.energy > 0:
            return False
        else:
            return True

    def smell_food(self, foods):
        sensory_spike_array = np.zeros(4)
        closest_food = min(foods, key= lambda food: np.linalg.norm(self.position - food.position))

        if self.position[0] < closest_food.position[0]: sensory_spike_array[0] = 1
        elif self.position[0] > closest_food.position[0]: sensory_spike_array[1] = 1

        if self.position[1] < closest_food.position[1]: sensory_spike_array[2] = 1
        elif self.position[1] > closest_food.position[1]: sensory_spike_array[3] = 1

        return sensory_spike_array
    
    def eat_food(self, foods):
        for food in foods:
            if np.array_equal(self.position, food.position):
                self.energy = min(self.energy + 20, 30)
                return food
        return None

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
    def __init__(self, world_size=(0,20,0,20)):
        self.position = np.random.randint(world_size[0], world_size[1], size=2)
        self.image = FOOD_IMG
        self.zoom = .15

class Home():
    def __init__(self, world_size=(0,20,0,20)):
        self.position = np.random.randint(world_size[0], world_size[1], size=2)
        self.image = HOME_IMG
        self.zoom = 0.2

def generate_genome_no_connections(num_input_neurons, num_output_neurons):
    total_neuron_count = num_input_neurons + num_output_neurons
    genome = Genome()
    activation_thresholds = [1 for _ in range(total_neuron_count)]
    leaks = [1 - random.uniform(.1, .3) for _ in range(total_neuron_count)]
    for i in range(num_input_neurons):
        genome.add_neuron("sensory", activation_thresholds.pop(0), leaks.pop(0))
    for i in range(num_output_neurons):
        genome.add_neuron("motor", activation_thresholds.pop(0), leaks.pop(0))
    return genome

def type_compatible(neuron_source, neuron_target) -> bool: 
    match neuron_source.type:
        case "hidden": 
            return True
        case "motor":
            if neuron_target.type == "motor":
                return False
            else:
                return True
        case "sensory":
            if neuron_target.type == "sensory":
                return False
            else:
                return True
        case _:
            raise Exception("no or wrong type given for neuron")

def transmitter_weight_positive_or_negative(transmitter) -> int:
    if transmitter == "excitatory":
        return 1
    else:
        return -1

def mutate_genome(genome) -> Genome:
    probability = random.random()
    get_neuron_key = lambda: random.choice(list(genome.neurons.keys()))

    if probability <= 0.30:
        transmitter = random.choices(["inhibitory", "excitatory"], weights=[1, 9])
        activation_threshold = random.randint(2, 8)
        leak = 1 - random.uniform(.1, .3)
        genome.add_neuron("hidden", activation_threshold, leak, transmitter)
    elif probability <= 0.70:
        compatible = False
        while(not compatible):
            (neuron_a_key, neuron_a_gene), (neuron_b_key, neuron_b_gene) = random.sample(list(genome.neurons.items()), 2)
            compatible = type_compatible(neuron_a_gene, neuron_b_gene)
        weight = random.randint(1, 3) * transmitter_weight_positive_or_negative(neuron_a_gene.transmitter)
        genome.add_synapse(neuron_a_key, neuron_b_key, weight)
    elif probability <= 0.85:
        genome_keys = list(genome.synapses.keys())
        if not genome_keys:
            return genome
        synapse_key = random.choice(genome_keys)
        new_weight = math.copysign(random.randint(1, 2), genome.synapses[synapse_key]["weight"])
        genome.synapses[synapse_key]["weight"] += new_weight
    elif probability <= 0.90:
        neuron_key = get_neuron_key()
        leak = random.uniform(-.1, .1)
        genome.neurons[neuron_key]["leak"] = max(genome.neurons[neuron_key]["leak"] + leak, .7)
    else:
        neuron_key = get_neuron_key()
        delta = random.randint(-2, 2)
        genome.neurons[neuron_key]["activation_threshold"] = max(genome.neurons[neuron_key]["activation_threshold"] + delta, 1)
    return genome

class World():
    def __init__(self, world_size=(0, 20, 0, 20)):
        self.fig, self.ax = plt.subplots(1, 1)
        plt.ion()
        self.world_size = world_size

    def update_plot(self, objects):
        ax = self.ax
        ax.cla()
        ax.axis(self.world_size)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        for obj in objects:
            ab = AnnotationBbox(OffsetImage(obj.image, zoom=obj.zoom),
                                (obj.position[0], obj.position[1]),
                                frameon=False)
            ax.add_artist(ab)
        plt.pause(.2)

class Simulation():
    def __init__(self, num_creatures=1, num_foods=3, num_homes=1, generations=10, time_steps=20, proportion_lost_each_generation=0.5):
        self.world = World(world_size=(0, 100, 0, 100))
        self.num_creatures = num_creatures
        if self.num_creatures % 2 == 1:
            raise Exception('Number of Creatures must be even.')
        self.num_foods = num_foods
        self.num_homes = num_homes
        self.creatures = [Creature(generate_genome_no_connections(4, 4), world_size=self.world.world_size) for _ in range(num_creatures)]
        self.foods = [Food(world_size=self.world.world_size) for _ in range(num_foods)]
        self.homes = [Home(world_size=self.world.world_size) for _ in range(num_homes)]
        self.generations = generations
        self.time_steps = time_steps
        self.proportion_lost_each_generation = proportion_lost_each_generation
        self.fitness_tracker = {}
        
    def __simulate_creature(self, creature):
        foods = self.foods
        sensory_data = creature.smell_food(foods)
        eaten_food = creature.eat_food(foods)
        if eaten_food:
            foods = [food for food in foods if food is not eaten_food]
            self.fitness_tracker[creature] += 1
            return
        movement_data = creature.think(sensory_data)
        creature.move(movement_data)
    
    def __evolve_creatures(self, creatures=None):
        if not creatures:
            creatures = self.creatures
        for i in range(len(creatures)):
            creatures[i] = Creature(mutate_genome(creatures[i].genome), world_size=self.world.world_size)

    def __run_generation(self):
        objects = self.creatures + self.foods + self.homes
        self.world.update_plot(objects)
        for _ in range(self.time_steps):
            for i in range(len(self.creatures)):
                self.__simulate_creature(self.creatures[i])
            objects = self.creatures + self.foods + self.homes
            self.world.update_plot(objects)

    def __natural_selection(self):
        amount_needed = int(self.proportion_lost_each_generation * self.num_creatures)
        if len(self.fitness_tracker) != 0:
            fit_creatures = dict(sorted(self.fitness_tracker.items(), key=operator.itemgetter(1)))
        else:
            fit_creatures = {}
        fit_creatures = list(fit_creatures.keys())
        unfit_creatures = list(set(self.creatures) - set(fit_creatures))
        fit_creatures = unfit_creatures + fit_creatures
        self.creatures = fit_creatures[amount_needed:]

    def __breed_new_generation() -> list:
        pass

    def run_simulation(self):
        for _ in range(self.generations):
            self.__run_generation()
            self.__natural_selection()
            new_generation = self.__breed_new_generation()
            self.__evolve_creatures(creatures=new_generation)
            self.fitness_tracker = {}

def main():
    simulation = Simulation(num_creatures=20, num_foods=10)
    simulation.run_simulation()

main()
