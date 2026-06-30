import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random
import math
import operator
import copy
from collections import defaultdict
from dataclasses import dataclass

CREATURE_IMG = mpimg.imread('sprites/mouse.png')
FOOD_IMG     = mpimg.imread('sprites/apple.png')

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
class SynapseGene:
    weight: float
    genetic_marker = None
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

    def add_neuron(self, *args):
        if len(args) == 1 and isinstance(args[0], NeuronGene):
            gene = args[0]
        else:
            gene = NeuronGene(*args)
        neuron_id = self.next_neuron_id
        self.next_neuron_id += 1
        self.neurons[neuron_id] = gene
        return neuron_id


    def add_synapse(self, in_id, out_id, weight):
        self.synapses[(in_id, out_id)] = SynapseGene(weight)

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

    def step(self, sensory_spike_array, max_spikes=1000):
        queued_neurons = []
        output_spike_array = np.zeros(len(self.motor_neurons))
        spike_count = 0

        for i in range(len(self.sensory_neurons)):
            self.sensory_neurons[i].add_voltage(sensory_spike_array[i])
            if self.sensory_neurons[i].attempt_fire():
                self.propagate(self.sensory_neurons[i], queued_neurons)
                spike_count += 1

        while len(queued_neurons) != 0 and spike_count < max_spikes:
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

        if spike_count >= max_spikes:
            print("max spikes reached")

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

def generate_genome_no_connections(num_input_neurons=4, num_output_neurons=4):
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
    def __init__(self, num_creatures=1, num_foods=3, generations=10, time_steps=20, proportion_lost_each_generation=0.5):
        self.world = World(world_size=(0, 50, 0, 50))
        self.num_creatures = num_creatures
        if self.num_creatures % 2 == 1:
            raise Exception('Number of Creatures must be even.')
        self.num_foods = num_foods
        self.creatures = [Creature(generate_genome_no_connections(4, 4), world_size=self.world.world_size) for _ in range(num_creatures)]
        self.foods = [Food(world_size=self.world.world_size) for _ in range(num_foods)]
        self.generations = generations
        self.time_steps = time_steps
        self.genetic_marker_tracker = 0
        self.proportion_lost_each_generation = proportion_lost_each_generation
        self.fitness_tracker = defaultdict(int)
    
    def __determine_fitness(self, creature, *, ate_food: bool = False, novelty: float = 0.0) -> None:
        if ate_food:
            self.fitness_tracker[creature] += 1

    def __simulate_creature(self, creature):
        sensory_data = creature.smell_food(self.foods)
        eaten_food = creature.eat_food(self.foods)
        if eaten_food:
            self.foods = [food for food in self.foods if food is not eaten_food]
        movement_data = creature.think(sensory_data)
        creature.move(movement_data)
        self.__determine_fitness(creature, ate_food=eaten_food)
    
    def __grow_and_evolve_creatures(self, creatures=None) -> list:
        new_creatures = []
        if not creatures:
            creatures = self.creatures
        for i in range(len(creatures)):
            new_creatures.append(Creature(mutate_genome(creatures[i].genome), world_size=self.world.world_size))
        return new_creatures

    def __run_generation(self):
        objects = self.creatures + self.foods
        self.world.update_plot(objects)
        for _ in range(self.time_steps):
            for i in range(len(self.creatures)):
                self.__simulate_creature(self.creatures[i])
            objects = self.creatures + self.foods
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

    def __clone_top_half_old_generation(self) -> list:
        return self.creatures[int(self.num_creatures * (self.proportion_lost_each_generation**2)):]

    def __is_type_hidden(self, neuron) -> bool:
        if neuron.type == "hidden":
            return True
        return False
    
    def __add_default_neurons(self, neuron_a_created_ids, neuron_b_created_ids, neuron_gene_id_translation, new_genome):
        for neuron_id in new_genome.neurons.keys():
            neuron_a_created_ids.append(neuron_id)
            neuron_b_created_ids.append(neuron_id)
            neuron_gene_id_translation[(1, neuron_id)] = neuron_id
            neuron_gene_id_translation[(2, neuron_id)] = neuron_id

    def __breed_new_generation(self, probability_for_fittest_genes=0.8) -> list:
        new_generation = []
        for i in range(0, len(self.creatures), 2) :
            creature_a = self.creatures[i]
            creature_b = self.creatures[i+1]
            creature_a_genome = creature_a.genome
            creature_b_genome = creature_b.genome
            new_genome = generate_genome_no_connections()
            neuron_gene_id_translation = {}
            synapse_a_created_ids = []
            synapse_b_created_ids = []
            neuron_a_created_ids = []
            neuron_b_created_ids = []

            self.__add_default_neurons(neuron_a_created_ids, neuron_b_created_ids, neuron_gene_id_translation, new_genome)

            for ((neuron_gene_a_1id, neuron_gene_a_2id), synapse_gene_a) in creature_a_genome.synapses.items():
                if synapse_gene_a.genetic_marker:
                    for ((neuron_gene_b_1id, neuron_gene_b_2id), synapse_gene_b) in creature_b_genome.synapses.items():
                        if synapse_gene_b.genetic_marker == synapse_gene_a.genetic_marker:
                            is_hidden_1 = self.__is_type_hidden(creature_b_genome.neurons[neuron_gene_b_1id])
                            if neuron_gene_b_1id not in neuron_b_created_ids and is_hidden_1:
                                new_neuron_gene_1_id = new_genome.add_neuron(copy.deepcopy(creature_b.genome.neurons[neuron_gene_b_1id]))
                                neuron_gene_id_translation[(1, neuron_gene_b_1id)] = new_neuron_gene_1_id
                                neuron_a_created_ids.append(neuron_gene_a_1id)
                                neuron_b_created_ids.append(neuron_gene_b_1id)
                            elif not is_hidden_1:
                                new_neuron_gene_1_id = neuron_gene_b_1id
                            else:
                                new_neuron_gene_1_id = neuron_gene_id_translation[(1, neuron_gene_b_1id)]
                            is_hidden_2 = self.__is_type_hidden(creature_b_genome.neurons[neuron_gene_b_2id])
                            if neuron_gene_b_2id not in neuron_b_created_ids and is_hidden_2:
                                new_neuron_gene_2_id = new_genome.add_neuron(copy.deepcopy(creature_b.genome.neurons[neuron_gene_b_2id]))
                                neuron_gene_id_translation[(2, neuron_gene_b_2id)] = new_neuron_gene_2_id
                                neuron_a_created_ids.append(neuron_gene_a_2id)
                                neuron_b_created_ids.append(neuron_gene_b_2id)
                            elif not is_hidden_2:
                                new_neuron_gene_2_id = neuron_gene_b_2id
                            else:
                                new_neuron_gene_2_id = neuron_gene_id_translation[(2, neuron_gene_b_2id)]
                            synapse_a_created_ids.append((neuron_gene_a_1id, neuron_gene_a_2id))
                            synapse_b_created_ids.append((neuron_gene_b_1id, neuron_gene_b_2id))
                            new_genome.add_synapse(new_neuron_gene_1_id, new_neuron_gene_2_id, synapse_gene_b.weight)

            synapse_a_uncreated_ids = list(set(creature_a_genome.synapses.keys()) - set(synapse_a_created_ids))
            synapse_b_uncreated_ids = list(set(creature_b_genome.synapses.keys()) - set(synapse_b_created_ids))
            neuron_a_uncreated_ids = list(set(creature_a_genome.neurons.keys()) - set(neuron_a_created_ids))
            neuron_b_uncreated_ids = list(set(creature_b_genome.neurons.keys()) - set(neuron_b_created_ids))
            
            b_gene_probability = 0.50
            same_fitnesss = self.fitness_tracker.get(creature_a) == self.fitness_tracker.get(creature_b)
            if not same_fitnesss and self.fitness_tracker.get(creature_b) != None:
                b_gene_probability = probability_for_fittest_genes
            new_synapses_a = [(key, creature_a_genome.synapses[key]) for key in synapse_a_uncreated_ids if random.random() <= 1 - b_gene_probability]
            new_synapses_b = [(key, creature_b_genome.synapses[key]) for key in synapse_b_uncreated_ids if random.random() <= b_gene_probability]
            for ((id_1, id_2), synapse_gene_a) in new_synapses_a:
                neuron_a = creature_a_genome.neurons[id_1]
                neuron_b = creature_a_genome.neurons[id_2]
                if id_1 in neuron_a_uncreated_ids:
                    neuron_a_id = new_genome.add_neuron(copy.deepcopy(neuron_a))
                    neuron_gene_id_translation[(1, id_1)] = neuron_a_id
                    neuron_a_uncreated_ids.remove(id_1)
                else:
                    neuron_a_id = neuron_gene_id_translation[(1, id_1)]
                if id_2 in neuron_a_uncreated_ids:
                    neuron_b_id = new_genome.add_neuron(copy.deepcopy(neuron_b))
                    neuron_gene_id_translation[(1, id_2)] = neuron_b_id
                    neuron_a_uncreated_ids.remove(id_2)
                else:
                    neuron_b_id = neuron_gene_id_translation[(1, id_2)]
                new_genome.add_synapse(neuron_a_id, neuron_b_id, synapse_gene_a.weight)
                new_genome.synapses[(neuron_a_id, neuron_b_id)].genetic_marker = self.genetic_marker_tracker
                self.genetic_marker_tracker += 1

            for ((id_1, id_2), synapse_gene_b) in new_synapses_b:
                neuron_a = creature_b_genome.neurons[id_1]
                neuron_b = creature_b_genome.neurons[id_2]
                if id_1 in neuron_b_uncreated_ids:
                    neuron_a_id = new_genome.add_neuron(copy.deepcopy(neuron_a))
                    neuron_gene_id_translation[(2, id_1)] = neuron_a_id
                    neuron_b_uncreated_ids.remove(id_1)
                else:
                    neuron_a_id = neuron_gene_id_translation[(2, id_1)]
                if id_2 in neuron_b_uncreated_ids:
                    neuron_b_id = new_genome.add_neuron(copy.deepcopy(neuron_b))
                    neuron_gene_id_translation[(2, id_2)] = neuron_b_id
                    neuron_b_uncreated_ids.remove(id_2)
                else:
                    neuron_b_id = neuron_gene_id_translation[(2, id_2)]
                new_genome.add_synapse(neuron_a_id, neuron_b_id, synapse_gene_b.weight)
                new_genome.synapses[(neuron_a_id, neuron_b_id)].genetic_marker = self.genetic_marker_tracker
                self.genetic_marker_tracker += 1
            
            for neuron_id in neuron_a_uncreated_ids:
                neuron_genome = creature_a_genome.neurons[neuron_id]
                if neuron_genome.type == "hidden" and random.random() >= 0.2:
                    new_genome.add_neuron(copy.deepcopy(neuron_genome))
               

            for neuron_id in neuron_b_uncreated_ids:
                neuron_genome = creature_b_genome.neurons[neuron_id]
                if neuron_genome.type != "hidden":
                    new_genome.add_neuron(copy.deepcopy(neuron_genome))
                elif random.random() >= 0.2:
                    new_genome.add_neuron(copy.deepcopy(neuron_genome))

            
            new_generation.append(Creature(new_genome))
        return new_generation

    def run_simulation(self):
        for _ in range(self.generations):
            self.__run_generation()
            self.__natural_selection()
            clone_generation = self.__clone_top_half_old_generation()
            new_generation = self.__breed_new_generation()
            new_creatures = self.__grow_and_evolve_creatures(creatures=clone_generation + new_generation)
            self.creatures += new_creatures
            self.fitness_tracker = defaultdict(int)

def main():
    simulation = Simulation(num_creatures=20, num_foods=20, generations=400)
    simulation.run_simulation()

main()
