from time import time
from search import *
from assignment1aux import *

def read_initial_state_from_file(filename):
    # Task 1
    # Return an initial state constructed using a configuration in a file.
    # Replace the line below with your code.
    with open(filename, 'r') as file:
        lines = file.readlines()
        height = int(lines[0].strip())
        width = int(lines[1].strip())
        rocks = [(int(x.split(',')[0]), int(x.split(',')[1])) for x in lines[2:]]
        
        # Initialize the map with empty strings.
        garden_map = [['' for _ in range(width)] for _ in range(height)]
        
        # Place rocks on the map.
        for rock in rocks:
            garden_map[rock[0]][rock[1]] = 'rock'
        
        # Convert the garden map and rows to tuples to make them immutable.
        garden_map = tuple(tuple(row) for row in garden_map)
        
        return (garden_map, None, None)

class ZenPuzzleGarden(Problem):
    def __init__(self, initial):
        if type(initial) is str:
            super().__init__(read_initial_state_from_file(initial))
        else:
            super().__init__(initial)

    def actions(self, state):
        # Task 2.1
        # Return a list of all allowed actions in a given state.
        garden_map, position, direction = state
        actions = []
        if position is None:  # Agent is outside the map
            for i, row in enumerate(garden_map):
                for j, tile in enumerate(row):
                    if tile == '':  # Unraked tile
                        if i == 0: actions.append(((i, j), 'down'))  # Top edge
                        if i == len(garden_map) - 1: actions.append(((i, j), 'up'))  # Bottom edge
                        if j == 0: actions.append(((i, j), 'right'))  # Left edge
                        if j == len(row) - 1: actions.append(((i, j), 'left'))  # Right edge
        else:  # Agent is inside the map
            if direction in ['left', 'right']:
                possible_directions = [('up', (position[0] - 1, position[1])), ('down', (position[0] + 1, position[1]))]
            else:
                possible_directions = [('left', (position[0], position[1] - 1)), ('right', (position[0], position[1] + 1))]
            for dir, new_pos in possible_directions:
                if 0 <= new_pos[0] < len(garden_map) and 0 <= new_pos[1] < len(garden_map[0]) and garden_map[new_pos[0]][new_pos[1]] == '':
                    actions.append((position, dir))
        return actions

    def result(self, state, action):
        # Task 2.2
        # Return a new state resulting from a given action being applied to a given state.
        garden_map, _, _ = state
        new_map = [list(row) for row in garden_map]  # Convert map to list for mutability
        position, direction = action
        # Initialize position for agents entering the map
        if position: 
            new_map[position[0]][position[1]] = direction  # Mark the entry tile as raked
            next_position = self._move(position, direction)
            while self._is_valid(new_map, next_position):
                r, c = next_position
                new_map[r][c] = direction  # Rake the tile
                next_position = self._move(next_position, direction)
        return (tuple(tuple(row) for row in new_map), None, None)  # Convert map back to tuple

    def goal_test(self, state):
        # Task 2.3
        # Return a boolean value indicating if a given state is solved.
        garden_map, position, _ = state
        return all(tile in ['rock', 'left', 'right', 'up', 'down'] for row in garden_map for tile in row) and position is None
    
    def _move(self, position, direction):
        """Compute new position based on direction."""
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        return (position[0] + moves[direction][0], position[1] + moves[direction][1])

    def _is_valid(self, garden_map, position):
        """Check if position is within bounds and not a rock."""
        if 0 <= position[0] < len(garden_map) and 0 <= position[1] < len(garden_map[0]):
            return garden_map[position[0]][position[1]] in ['', 'rock']  # Consider rocks as stopping points
        return False
# Task 3
# Implement an A* heuristic cost function and assign it to the variable below.
def heuristic(node):

    state = node.state[0]  # Extract the garden map from the state
    unraked_tiles = sum(1 for row in state for tile in row if tile == '')
    return unraked_tiles

# Assign the heuristic function to the variable for use in A* search
astar_heuristic_cost = heuristic

def beam_search(problem, f, beam_width):
    # Task 4
    # Implement a beam-width version A* search.
    # Return a search node containing a solved state.
    # Experiment with the beam width in the test code to find a solution.
    # Initialize the frontier with the initial state of the problem.
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    
    # Frontier is a list of nodes; initialize with the starting node.
    frontier = [node]
    
    # Loop until a solution is found or the frontier is empty.
    while frontier:
        # Expand the frontier and keep only the best beam_width nodes based on their f cost.
        successors = []
        for node in frontier:
            # Expand the current node to obtain successors.
            for action in problem.actions(node.state):
                child = node.child_node(problem, action)
                if problem.goal_test(child.state):
                    return child
                successors.append(child)
        
        # Keep only beam_width nodes with the lowest f cost.
        frontier = sorted(successors, key=f)[:beam_width]
    
    # Return None if no solution was found within the beam width.
    return None

if __name__ == "__main__":

    # Task 1 test code
    
    print('The loaded initial state is visualised below.')
    visualise(read_initial_state_from_file('assignment1config.txt'))
    

    # Task 2 test code
    
    garden = ZenPuzzleGarden('assignment1config.txt')
    print('Running breadth-first graph search.')
    before_time = time()
    node = breadth_first_graph_search(garden)
    after_time = time()
    print(f'Breadth-first graph search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
    

    # Task 3 test code
    
    print('Running A* search.')
    before_time = time()
    node = astar_search(garden, astar_heuristic_cost)
    after_time = time()
    print(f'A* search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
    

    # Task 4 test code
    
    print('Running beam search.')
    before_time = time()
    node = beam_search(garden, lambda n: n.path_cost + astar_heuristic_cost(n), 50)
    after_time = time()
    print(f'Beam search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
    
