from search import *                     # AIMA search algorithms
from random import randint
from KNetWalk_Optimisationaux import *   # Helper functions for visualisation


def read_tiles_from_file(filename):
    """
    Reads a configuration file and returns the KNetWalk board
    in its tile-connection representation.

    File encoding:
        ' ' → empty tile ()
        'i' → terminal tile (0,)
        'L' → L-shaped tile (0, 1)
        'I' → straight tile (0, 2)
        'T' → T-shaped tile (0, 1, 2)

    Args:
        filename (str): path to configuration file

    Returns:
        tuple: nested tuple representing the board
    """
    tile_mapping = {
        ' ': (),         # Empty
        'i': (0,),       # Terminal
        'L': (0, 1),     # L-shape
        'I': (0, 2),     # Straight
        'T': (0, 1, 2),  # T-shape
    }

    board = []
    with open(filename, 'r') as file:
        for line in file:
            row = tuple(tile_mapping.get(char, ()) for char in line.strip('\n'))
            board.append(row)
    return tuple(board)


class KNetWalk(Problem):
    """
    KNetWalk puzzle implemented as a search/optimisation problem.
    Tiles must be rotated until all connections form a single network.
    """

    def __init__(self, tiles):
        # Accept either a filename or pre-defined board
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles

        height = len(self.tiles)
        width = len(self.tiles[0])

        # Maximum fitness = fully solved network
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)

        # Random initial state for optimisation algorithms
        super().__init__(self.generate_random_state())

    def generate_random_state(self):
        """Generates a random orientation (0–3) for each tile."""
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]

    def actions(self, state):
        """
        Returns all valid actions for a state.

        Each action = (row, col, new_orientation)
        """
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [(i, j, k)
                for i in range(height)
                for j in range(width)
                for k in [0, 1, 2, 3]
                if state[i * width + j] != k]

    def result(self, state, action):
        """Applies an action and returns a new state (list of orientations)."""
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]

    def goal_test(self, state):
        """Checks if the state reaches maximum fitness."""
        return self.value(state) == self.max_fitness

    def value(self, state):
        """
        Computes the fitness of a state.
        +2 points for each valid connection between neighbouring tiles.
        """
        height = len(self.tiles)
        width = len(self.tiles[0])
        fitness = 0

        def oriented_connections(orientation, connections):
            """Rotate connections by orientation (mod 4)."""
            return tuple((connection + orientation) % 4 for connection in connections)

        # Evaluate each tile
        for i in range(height):
            for j in range(width):
                current_tile = self.tiles[i][j]
                current_orientation = state[i * width + j]
                connections = oriented_connections(current_orientation, current_tile)

                # Right neighbour
                if j < width - 1:
                    right_tile = self.tiles[i][j + 1]
                    right_orientation = state[i * width + j + 1]
                    right_connections = oriented_connections(right_orientation, right_tile)
                    if 0 in connections and 2 in right_connections:
                        fitness += 2

                # Bottom neighbour
                if i < height - 1:
                    bottom_tile = self.tiles[i + 1][j]
                    bottom_orientation = state[(i + 1) * width + j]
                    bottom_connections = oriented_connections(bottom_orientation, bottom_tile)
                    if 3 in connections and 1 in bottom_connections:
                        fitness += 2

                # Early exit if solved
                if fitness >= self.max_fitness:
                    return fitness

        return fitness


# -------------------------
# Optimisation Parameters
# -------------------------

# Simulated Annealing schedule
sa_schedule = exp_schedule(k=20, lam=0.005, limit=1000)

# Genetic Algorithm parameters
pop_size = 100
num_gen = 100
mutation_prob = 0.1


# -------------------------
# Beam Search Variants
# -------------------------

def local_beam_search(problem, population):
    """
    Local beam search: keeps the top-b fittest states
    at each step, where b = initial population size.
    """
    beam_width = len(population)

    while True:
        successors = []
        for state in population:
            for action in problem.actions(state):
                successors.append(problem.result(state, action))

        # Sort by fitness, keep best beam_width
        successors = sorted(successors, key=lambda s: problem.value(s), reverse=True)
        new_population = successors[:beam_width]

        # Stop if no improvement
        if new_population == population:
            break
        population = new_population

    return max(population, key=problem.value)


def stochastic_beam_search(problem, population, limit=1000):
    """
    Stochastic beam search:
    - Expands all successors
    - Selects next population using fitness-weighted sampling
    - Runs until solution found or iteration limit reached
    """
    import random
    beam_width = len(population)

    for _ in range(limit):
        successors = []
        for state in population:
            for action in problem.actions(state):
                successors.append(problem.result(state, action))

        # Weighted random selection
        weights = [problem.value(state) for state in successors]
        if sum(weights) == 0:
            population = random.sample(successors, beam_width)
        else:
            population = random.choices(successors, weights=weights, k=beam_width)

        # Stop early if goal found
        if any(problem.goal_test(state) for state in population):
            return max(population, key=problem.value)

    return max(population, key=problem.value)


# -------------------------
# Main Execution
# -------------------------

if __name__ == '__main__':

    # Load puzzle
    network = KNetWalk('KNetWalk_Optimisationconfig.txt')
    print("Initial puzzle state:")
    visualise(network.tiles, network.initial)

    # Hill Climbing
    run = 0
    while True:
        state = hill_climbing(network)
        if network.goal_test(state):
            print(f"\nHill Climbing solved on run {run}")
            visualise(network.tiles, state)
            break
        else:
            print(f"Hill Climbing run {run}: best fitness {network.value(state)}/{network.max_fitness}")
            run += 1

    # Simulated Annealing
    run = 0
    while True:
        state = simulated_annealing(network, schedule=sa_schedule)
        if network.goal_test(state):
            print(f"\nSimulated Annealing solved on run {run}")
            visualise(network.tiles, state)
            break
        else:
            print(f"Simulated Annealing run {run}: best fitness {network.value(state)}/{network.max_fitness}")
            run += 1

    # Genetic Algorithm
    run = 0
    while True:
        state = genetic_algorithm(
            [network.generate_random_state() for _ in range(pop_size)],
            network.value, [0, 1, 2, 3], network.max_fitness,
            num_gen, mutation_prob
        )
        if network.goal_test(state):
            print(f"\nGenetic Algorithm solved on run {run}")
            visualise(network.tiles, state)
            break
        else:
            print(f"Genetic Algorithm run {run}: best fitness {network.value(state)}/{network.max_fitness}")
            run += 1

    # Local Beam Search
    run = 0
    while True:
        state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            print(f"\nLocal Beam Search solved on run {run}")
            visualise(network.tiles, state)
            break
        else:
            print(f"Local Beam Search run {run}: best fitness {network.value(state)}/{network.max_fitness}")
            run += 1

    # Stochastic Beam Search
    run = 0
    while True:
        state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            print(f"\nStochastic Beam Search solved on run {run}")
            visualise(network.tiles, state)
            break
        else:
            print(f"Stochastic Beam Search run {run}: best fitness {network.value(state)}/{network.max_fitness}")
            run += 1
