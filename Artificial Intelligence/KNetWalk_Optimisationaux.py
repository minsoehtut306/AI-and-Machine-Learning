def visualise(tiles, state):
    """
    Prints a text-based visualisation of the KNetWalk board.

    Each tile is represented by Unicode box-drawing characters,
    rotated according to its orientation in the given state.

    Args:
        tiles (tuple): Nested tuple representing the board.
            Each element is a tuple of connections:
                ()        → empty tile
                (0,)      → terminal
                (0, 1)    → L-shaped
                (0, 2)    → straight
                (0, 1, 2) → T-shaped
        state (list): Flat list of orientations (0–3) for each tile.
            The (i * width + j)-th entry is the orientation of tile (i, j).

    Symbols used (depending on orientation):
        • Empty    : " " (space)
        • Terminal : ┐ └ ┘ ┌  (varies by orientation)
        • L-shaped : ┗ ┛ ┓ ┏
        • Straight : ━ ┃ (horizontal/vertical)
        • T-shaped : ┻ ┫ ┳ ┣
    """
    height = len(tiles)
    width = len(tiles[0])
    print()

    for i in range(height):
        for j in range(width):
            tile = tiles[i][j]
            orientation = state[i * width + j]

            if not tile:  # Empty
                print(' ', end='')

            elif tile == (0,):  # Terminal
                print(['\u257a', '\u2579', '\u2578', '\u257b'][orientation], end='')

            elif tile == (0, 1):  # L-shaped
                print(['\u2517', '\u251b', '\u2513', '\u250f'][orientation], end='')

            elif tile == (0, 2):  # Straight
                print(['\u2501', '\u2503', '\u2501', '\u2503'][orientation], end='')

            elif tile == (0, 1, 2):  # T-shaped
                print(['\u253b', '\u252b', '\u2533', '\u2523'][orientation], end='')

            else:
                # Unexpected tile encoding
                print()
                print(f"Unexpected tile representation encountered: {tile}.")
                print("Valid tiles are (), (0,), (0,1), (0,2), (0,1,2).")
                raise ValueError(tile)

        print()
    print()
