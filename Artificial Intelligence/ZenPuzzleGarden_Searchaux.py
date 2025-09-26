from os import system
from time import sleep

def visualise(state):
    """
    Prints a text-based visualisation of the garden state.

    Symbols:
        ▀ / ▄   : garden borders
        █       : rock
        ◂ ▴ ▸ ▾ : raked tiles (left, up, right, down)
        ⟠ / ⟡   : monk (depending on direction)
        ○       : monk without a direction
        (space) : unraked tile

    Args:
        state (tuple): (garden_map, position, direction)
            - garden_map: tuple of tuples representing tiles
            - position: agent's (row, col) or None if outside
            - direction: 'left'/'right'/'up'/'down' or None
    """
    garden_map, position, move = state
    height = len(garden_map)
    width = len(garden_map[0])

    # Top border
    print('\n ', *['\u2581' for _ in range(width)], ' ', sep='')

    for i in range(height):
        print('\u2595', end='')  # Left border
        for j in range(width):
            if garden_map[i][j] == 'rock':
                print('\u2588', end='')  # Rock
            elif garden_map[i][j] == 'left':
                print('\u25c2', end='')  # ◂
            elif garden_map[i][j] == 'up':
                print('\u25b4', end='')  # ▴
            elif garden_map[i][j] == 'right':
                print('\u25b8', end='')  # ▸
            elif garden_map[i][j] == 'down':
                print('\u25be', end='')  # ▾
            elif position == (i, j):  # Agent currently here
                if move == 'right':
                    print('\u2520', end='')  # ┠
                elif move == 'left':
                    print('\u2528', end='')  # ┨
                elif move == 'down':
                    print('\u252f', end='')  # ┯
                elif move == 'up':
                    print('\u2537', end='')  # ┷
                else:
                    print('\u25cb', end='')  # ○
            elif not garden_map[i][j]:  
                print(' ', end='')  # Empty/unraked tile
            else:
                # Catch unexpected tile types
                print()
                print(f"Unexpected tile type: '{garden_map[i][j]}'")
                print("Valid tiles: '', 'rock', 'left', 'right', 'up', 'down'")
                raise ValueError(garden_map[i][j])
        print('\u258f')  # Right border

    # Bottom border
    print(' ', *['\u2594' for _ in range(width)], ' \n', sep='')


def animate(node):
    """
    Animates the solution path step by step in the terminal.

    Args:
        node (Node): final search node containing the solution path.
                     Uses node.path() to iterate over states.
    """
    for node_i in node.path()[:-1]:
        visualise(node_i.state)
        sleep(2)  # Pause for readability
        # Move cursor up to redraw the grid in place
        for _ in range(len(node_i.state[0]) + 4):
            system('tput cuu1')

    # Show final state
    visualise(node.state)
