from time import time
from search import *                     # AIMA search algorithms (https://github.com/aimacode/aima-python)
from ZenPuzzleGarden_Searchaux import *  # Helper functions for visualisation & animation


def read_initial_state_from_file(filename):
    """
    Reads a garden configuration file and returns the initial state.

    Config file format:
    - Line 1: garden height (rows)
    - Line 2: garden width (columns)
    - Remaining lines: rock tile positions as "row,col"

    Returns:
        state (tuple): (garden_map, agent_position, agent_direction)
            - garden_map: tuple of tuples of strings
                '' (unraked), 'rock', 'left', 'right', 'up', 'down'
            - agent_position: None if outside, or (row, col) if inside
            - agent_direction: None if outside, or 'left'/'right'/'up'/'down'
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        height = int(lines[0].strip())
        width = int(lines[1].strip())
        rocks = [(int(x.split(',')[0]), int(x.split(',')[1])) for x in lines[2:]]

        # Create empty garden
        garden_map = [['' for _ in range(width)] for _ in range(height)]

        # Place rocks
        for rock in rocks:
            garden_map[rock[0]][rock[1]] = 'rock'

        # Convert to immutable structure (tuples) for hashing in search
        garden_map = tuple(tuple(row) for row in garden_map)

        # Agent starts outside (None, None)
        return (garden_map, None, None)


class ZenPuzzleGarden(Problem):
    """
    Puzzle environment:
    - Agent (monk) enters the garden from the perimeter
    - Must rake every unraked tile, leaving rocks untouched
    - Movement rules follow Zen Puzzle Garden mechanics
    """

    def __init__(self, initial):
        if type(initial) is str:
            super().__init__(read_initial_state_from_file(initial))
        else:
            super().__init__(initial)

    def actions(self, state):
        """
        Returns possible actions in the given state.
        Each action is represented as (position, direction).
        """
        garden_map, position, direction = state
        actions = []

        if position is None:  # Agent outside garden → must enter
            for i, row in enumerate(garden_map):
                for j, tile in enumerate(row):
                    if tile == '':  # Can only enter on unraked tiles
                        if i == 0: actions.append(((i, j), 'down'))     # Enter from top
                        if i == len(garden_map) - 1: actions.append(((i, j), 'up'))   # Enter from bottom
                        if j == 0: actions.append(((i, j), 'right'))   # Enter from left
                        if j == len(row) - 1: actions.append(((i, j), 'left'))  # Enter from right
        else:  # Agent inside garden → can only turn 90° left/right
            if direction in ['left', 'right']:
                possible_directions = [('up', (position[0] - 1, position[1])),
                                       ('down', (position[0] + 1, position[1]))]
            else:
                possible_directions = [('left', (position[0], position[1] - 1)),
                                       ('right', (position[0], position[1] + 1))]
            for dir, new_pos in possible_directions:
                if (0 <= new_pos[0] < len(garden_map) and
                    0 <= new_pos[1] < len(garden_map[0]) and
                    garden_map[new_pos[0]][new_pos[1]] == ''):
                    actions.append((position, dir))
        return actions

    def result(self, state, action):
        """
        Applies an action to the current state and returns a new state.
        The agent moves in the chosen direction until hitting a rock,
        a raked tile, or exiting the garden.
        """
        garden_map, _, _ = state
        new_map = [list(row) for row in garden_map]  # Copy as list for mutability
        position, direction = action

        if position: 
            # Mark entry tile
            new_map[position[0]][position[1]] = direction
            next_position = self._move(position, direction)

            # Continue moving until blocked or out of bounds
            while self._is_valid(new_map, next_position):
                r, c = next_position
                new_map[r][c] = direction  # Rake the tile
                next_position = self._move(next_position, direction)

        # Agent always ends outside (None, None) in this implementation
        return (tuple(tuple(row) for row in new_map), None, None)

    def goal_test(self, state):
        """
        Checks if the garden is fully raked (or rocks)
        and the agent has exited the garden.
        """
        garden_map, position, _ = state
        return (all(tile in ['rock', 'left', 'right', 'up', 'down']
                    for row in garden_map for tile in row)
                and position is None)

    def _move(self, position, direction):
        """Returns new position after moving one step in the given direction."""
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        return (position[0] + moves[direction][0], position[1] + moves[direction][1])

    def _is_valid(self, garden_map, position):
        """Checks if position is inside garden and not blocked by raked tile."""
        if 0 <= position[0] < len(garden_map) and 0 <= position[1] < len(garden_map[0]):
            return garden_map[position[0]][position[1]] in ['', 'rock']
        return False


# ------------------------
# Search Strategies
# ------------------------

def heuristic(node):
    """
    Heuristic for A*:
    Count the number of unraked tiles.
    Optimistic estimate of remaining moves.
    """
    state = node.state[0]
    unraked_tiles = sum(1 for row in state for tile in row if tile == '')
    return unraked_tiles


# Assign heuristic for A* search
astar_heuristic_cost = heuristic


def beam_search(problem, f, beam_width):
    """
    Beam search: variant of A* that limits frontier size.
    Keeps only the 'beam_width' lowest-cost nodes at each step.
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    
    frontier = [node]

    while frontier:
        successors = []
        for node in frontier:
            for action in problem.actions(node.state):
                child = node.child_node(problem, action)
                if problem.goal_test(child.state):
                    return child
                successors.append(child)
        
        # Keep only best beam_width nodes
        frontier = sorted(successors, key=f)[:beam_width]
    
    return None  # No solution found


# ------------------------
# Main Execution
# ------------------------

if __name__ == "__main__":
    # Load initial configuration
    print('Initial garden state:')
    visualise(read_initial_state_from_file('ZenPuzzleGarden_Searchconfig.txt'))

    # Breadth-First Search
    garden = ZenPuzzleGarden('ZenPuzzleGarden_Searchconfig.txt')
    print('\nRunning Breadth-First Search...')
    before_time = time()
    node = breadth_first_graph_search(garden)
    after_time = time()
    print(f'BFS completed in {after_time - before_time:.4f} seconds.')
    if node:
        print(f'Solution found with cost {node.path_cost}.')
        animate(node)

    # A* Search
    print('\nRunning A* Search...')
    before_time = time()
    node = astar_search(garden, astar_heuristic_cost)
    after_time = time()
    print(f'A* completed in {after_time - before_time:.4f} seconds.')
    if node:
        print(f'Solution found with cost {node.path_cost}.')
        animate(node)

    # Beam Search
    print('\nRunning Beam Search...')
    before_time = time()
    node = beam_search(garden, lambda n: n.path_cost + astar_heuristic_cost(n), 50)
    after_time = time()
    print(f'Beam search completed in {after_time - before_time:.4f} seconds.')
    if node:
        print(f'Solution found with cost {node.path_cost}.')
        animate(node)
