from search import *
from random import randint
from assignment2aux import *


def read_tiles_from_file(filename):
    # Task 1
    # Return a tile board constructed using a configuration in a file.
    # Define a mapping of file characters to their tuple representations
    tile_mapping = {
        ' ': (),         # Empty tile
        'i': (0,),       # Terminal tile
        'L': (0, 1),     # L-shaped tile
        'T': (0, 1, 2),  # T-shaped tile

    }
    # Reading the file and constructing the board configuration.
    board = []
    with open(filename, 'r') as file:
        for line in file:
            row = tuple(tile_mapping.get(char, ()) for char in line.strip('\n'))
            board.append(row)
    return tuple (board)

class KNetWalk(Problem):
    def __init__(self, tiles):
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles
        height = len(self.tiles)
        width = len(self.tiles[0])
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)
        super().__init__(self.generate_random_state())


    def generate_random_state(self):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]


    def actions(self, state):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [(i, j, k) for i in range(height) for j in range(width) for k in [0, 1, 2, 3] if state[i * width + j] != k]


    def result(self, state, action):
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]


    def goal_test(self, state):
        return self.value(state) == self.max_fitness


    def value(self, state):
        # Task 2
        # Return an integer fitness value of a given state.
        height = len(self.tiles)
        width = len(self.tiles[0])
        fitness = 0

        # Helper function to calculate the effective connections of a tile
        def oriented_connections(orientation, connections):
            return tuple((connection + orientation) % 4 for connection in connections)

        # Check connections between tiles
        for i in range(height):
            for j in range(width):
                current_tile = self.tiles[i][j]
                current_orientation = state[i * width + j]
                connections = oriented_connections(current_orientation, current_tile)

                # Check right connection
                if j < width - 1:
                    right_tile = self.tiles[i][j + 1]
                    right_orientation = state[i * width + j + 1]
                    right_connections = oriented_connections(right_orientation, right_tile)
                    if 0 in connections and 2 in right_connections:
                        fitness += 2  # Increment by 2 for a valid connection to the right

                # Check bottom connection
                if i < height - 1:
                    bottom_tile = self.tiles[i + 1][j]
                    bottom_orientation = state[(i + 1) * width + j]
                    bottom_connections = oriented_connections(bottom_orientation, bottom_tile)
                    if 3 in connections and 1 in bottom_connections:
                        fitness += 2  # Increment by 2 for a valid connection downwards

                # Early stopping if maximum fitness is reached
                if fitness >= 20:
                    return fitness

        return fitness

# Task 3
# Configure an exponential schedule for simulated annealing.
sa_schedule = exp_schedule(k=20, lam=0.005, limit=1000)

# Task 4
# Configure parameters for the genetic algorithm.
pop_size = 100
num_gen = 100
mutation_prob = 0.1

def local_beam_search(problem, population):
    # Task 5
    # Implement local beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the next population contains no fitter state.
    beam_width = len(population)  # Keep the population size consistent.

    while True:
        successors = []
        for state in population:
            actions = problem.actions(state)  # Get available actions for the state
            for action in actions:
                successor = problem.result(state, action)  # Get result of applying action
                successors.append(successor)

        # Evaluate fitness of all successors
        successors = sorted(successors, key=lambda state: problem.value(state), reverse=True)
        
        # Select the top 'beam_width' successors for the next generation
        new_population = successors[:beam_width]
        
        # Check if there's any improvement
        if new_population == population:
            break
        population = new_population

    # Return the state with the highest fitness from the final population
    return max(population, key=problem.value)

def stochastic_beam_search(problem, population, limit=1000):
    # Task 6
    # Implement stochastic beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the generation limit is reached.
    beam_width = len(population)  # Keep the population size consistent.
    for _ in range(limit):
        # Generate all successors from the current population
        successors = []
        for state in population:
            successors.extend(problem.successors(state))  # Assuming 'successors' method returns all possible next states

        # Fitness-weighted selection of the new population
        weights = [problem.value(state) for state in successors]
        if sum(weights) == 0:  # Handling a potential division by zero in random.choices()
            population = random.sample(successors, beam_width)
        else:
            population = random.choices(successors, weights=weights, k=beam_width)

    # Return the state with the highest fitness from the final population
    return max(population, key=problem.value)


if __name__ == '__main__':

    # Task 1 test code
    
    network = KNetWalk('assignment2config.txt')
    visualise(network.tiles, network.initial)
    

    # Task 2 test code
    
    run = 0
    method = 'hill climbing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = hill_climbing(network)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 3 test code
    
    run = 0
    method = 'simulated annealing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = simulated_annealing(network, schedule=sa_schedule)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 4 test code
    
    run = 0
    method = 'genetic algorithm'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = genetic_algorithm([network.generate_random_state() for _ in range(pop_size)], network.value, [0, 1, 2, 3], network.max_fitness, num_gen, mutation_prob)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 5 test code
    
    run = 0
    method = 'local beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 6 test code
    
    run = 0
    method = 'stochastic beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    
