import math

from copy import deepcopy
from games4e import *

class EinStein(Game):

    def __init__(self):
        self.initial = GameState(to_move='R', utility=0, board={'R': (0, 0), 'B': (2, 2)}, moves=[(1, 1), (1, 0), (0, 1)])

    def compute_moves(self, board, to_move):
        moves = []
        if board[to_move]:
            if to_move == 'R':
                if board[to_move][0] < 2:
                    moves.append((board[to_move][0] + 1, board[to_move][1]))
                    if board[to_move][1] < 2:
                        moves.append((board[to_move][0] + 1, board[to_move][1] + 1))
                if board[to_move][1] < 2:
                    moves.append((board[to_move][0], board[to_move][1] + 1))
            if to_move == 'B':
                if board[to_move][0] > 0:
                    moves.append((board[to_move][0] - 1, board[to_move][1]))
                    if board[to_move][1] > 0:
                        moves.append((board[to_move][0] - 1, board[to_move][1] - 1))
                if board[to_move][1] > 0:
                    moves.append((board[to_move][0], board[to_move][1] - 1))
        return moves

    def display(self, state):
        displayed_board = [[' ' for _ in range(3)] for _ in range(3)]
        for player_i in ['R', 'B']:
            if state.board[player_i] is not None:
                displayed_board[state.board[player_i][0]][state.board[player_i][1]] = f'{player_i}'
        print('\n'.join(['|' + '|'.join(row) + '|' for row in displayed_board]), end='\n\n')

    def terminal_test(self, state):
        return state.utility != 0

    def actions(self, state):
        return state.moves

    def result(self, state, move):
        # Task 1.1
        # Return a state resulting from the move.
        # Replace the line below with your code.

        # Determine player and opponent
        player = state.to_move
        opponent = 'B' if player == 'R' else 'R'
        board_copy = deepcopy(state.board)
        # Execute the move
        board_copy[player] = move
        if move == board_copy[opponent]:
            board_copy[opponent] = None
        # Compute new utility and moves
        new_utility = self.compute_utility(board_copy)
        new_moves = self.compute_moves(board_copy, opponent)
        # Return new game state with opponent's turn
        return GameState(to_move=opponent, utility=new_utility, board=board_copy, moves=new_moves)

    def utility(self, state, player):
        # Task 1.2
        # Return the state's utility to the player.
        # Replace the line below with your code.
        
        # Return the stored utility if player is 'R', otherwise return its negated value if player is 'B'
        return state.utility if player == 'R' else -state.utility
    
    def compute_utility(self, board):
        # Task 1.3
        # Return the utility of the board.
        # Replace the line below with your code.
        
        # Check if Red's piece is removed or Blue's piece reaches the top left tile
        if board['R'] is None or board['B'] == (0, 0):
            return -1
        # Check if Blue's piece is removed or Red's piece reaches the bottom right tile
        elif board['B'] is None or board['R'] == (2, 2):
            return 1
        else:
            return 0

class MehrSteine(StochasticGame):

    def __init__(self, board_size):
        self.board_size = board_size
        self.num_piece = int((board_size - 1) * (board_size - 2) / 2)
        board = {'R': [], 'B': []}
        for i in range(board_size - 2):
            for j in range(board_size - 2 - i):
                board['R'].append((i, j))
                board['B'].append((board_size - 1 - i, board_size - 1 - j))
        self.initial = StochasticGameState(to_move='R', utility=0, board=board, moves=None, chance=None)

    def compute_moves(self, board, to_move, index):
        moves = []
        coordinates = board[to_move][index]
        if to_move == 'R':
            if coordinates[0] < self.board_size - 1:
                moves.append((index, (coordinates[0] + 1, coordinates[1])))
                if coordinates[1] < self.board_size - 1:
                    moves.append((index, (coordinates[0] + 1, coordinates[1] + 1)))
            if coordinates[1] < self.board_size - 1:
                moves.append((index, (coordinates[0], coordinates[1] + 1)))
        if to_move == 'B':
            if coordinates[0] > 0:
                moves.append((index, (coordinates[0] - 1, coordinates[1])))
                if coordinates[1] > 0:
                    moves.append((index, (coordinates[0] - 1, coordinates[1] - 1)))
            if coordinates[1] > 0:
                moves.append((index, (coordinates[0], coordinates[1] - 1)))
        return moves

    def display(self, state):
        spacing = 1 if self.num_piece == 1 else math.floor(math.log(self.num_piece - 1, 10)) + 1
        displayed_board = [[' ' * (spacing + 1) for _ in range(self.board_size)] for _ in range(self.board_size)]
        for player_i in ['R', 'B']:
            for piece_i in range(self.num_piece):
                if state.board[player_i][piece_i] is not None:
                    displayed_board[state.board[player_i][piece_i][0]][state.board[player_i][piece_i][1]] = player_i + str(piece_i).rjust(spacing)
        print('\n'.join(['|' + '|'.join(row) + '|' for row in displayed_board]), end='\n\n')

    def terminal_test(self, state):
        return state.utility != 0

    def actions(self, state):
        return state.moves

    def result(self, state, move):
        # Task 2.1
        # Return a state resulting from the move.
        # Replace the line below with your code.

        board_copy = deepcopy(state.board)
        current_player = state.to_move
        opponent = 'R' if current_player == 'B' else 'B'
        # Move details
        piece_index, destination = move
        current_position = board_copy[current_player][piece_index]
        # Execute the move
        board_copy[current_player][piece_index] = destination
        # Check for capture
        for idx, pos in enumerate(board_copy[opponent]):
            if pos == destination:
                board_copy[opponent][idx] = None  # Capture the opponent's piece
        # Prepare the new state
        new_to_move = opponent
        new_utility = self.compute_utility(board_copy)  
        # Construct and return the new game state
        return StochasticGameState(to_move=new_to_move, utility=new_utility, board=board_copy, moves=None, chance=None)

    def utility(self, state, player):
        # Task 2.2
        # Return the state's utility to the player.
        # Replace the line below with your code.
        
        # Return the stored utility if player is 'R', otherwise return its negated value if player is 'B'
        return state.utility if player == 'R' else -state.utility

    def compute_utility(self, board):
        # Task 2.3
        # Return the utility of the board.
        # Replace the line below with your code.

        top_left = (0, 0)  # Red's starting position
        bottom_right = (self.board_size - 1, self.board_size - 1)  # Blue's starting position
        # Check if Red loses all pieces or Blue reaches the top left tile
        if all(pos is None for pos in board['R']) or any(pos == top_left for pos in board['B']):
            return -1
        # Check if Blue loses all pieces or Red reaches the bottom right tile
        if all(pos is None for pos in board['B']) or any(pos == bottom_right for pos in board['R']):
            return 1
        return 0

    def chances(self, state):
        # Task 2.4
        # Return a list of possible chance outcomes.
        # Replace the line below with your code.
        
        #Return a list of integers from 0 to num_piece-1
        return list(range(self.num_piece))

    def outcome(self, state, chance):
        # Task 2.5
        # Return a state resulting from the chance outcome.
        # Replace the line below with your code.

        pieces = state.board[state.to_move]
        piece_position = pieces[chance] if chance < len(pieces) else None
        # Find a valid piece if the selected one is not available
        if piece_position is None:
            # Look for the next available piece below or above the chance index
            indices = list(range(len(pieces)))
            lower_indices = indices[:chance][::-1]
            upper_indices = indices[chance+1:]
            # Check for the nearest available pieces in both directions
            for idx in lower_indices + upper_indices:
                if pieces[idx] is not None:
                    piece_position = pieces[idx]
                    chance = idx
                    break
        # Compute moves for the determined piece
        if piece_position is not None:
            legal_moves = self.compute_moves(state.board, state.to_move, chance)
        else:
            legal_moves = []
        # Return new state with updated moves and the same other attributes
        return StochasticGameState(
            to_move=state.to_move,
            utility=state.utility,
            board=state.board,
            moves=legal_moves,
            chance=chance
        )

    def probability(self, chance):
        # Task 2.6
        # Return the probability of a chance outcome.
        # Replace the line below with your code.
        
        # Return the uniform probability of selecting any one piece
        return 1 / self.num_piec

def stochastic_monte_carlo_tree_search(state, game, playout_policy, N=1000):

    def select(n):
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.outcome(game.result(n.state, action), chance), parent=n): action for action in game.actions(n.state) for chance in game.chances(game.result(n.state, action))}
        return select(n)

    def simulate(game, state):
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = playout_policy(game, state)
            state = game.result(state, action)
            chance = random.choice(game.chances(state))
            state = game.outcome(state, chance)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        if utility > 0:
            n.U += utility
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)

    for _ in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)

def schwarz_score(game, state):
    schwarz = {}
    valid_pieces = [piece_i for piece_i in range(game.num_piece) if state.board['R'][piece_i] is not None]
    if len(valid_pieces) == 0:
        schwarz['R'] = (game.board_size - 1) * game.num_piece
    elif len(valid_pieces) == 1:
        schwarz['R'] = game.board_size - 1 - min(state.board['R'][valid_pieces[0]])
    else:
        schwarz_per_piece = []
        for index_i, piece_i in enumerate(valid_pieces):
            if index_i == 0:
                schwarz_per_piece.append((game.board_size - 1 - min(state.board['R'][piece_i])) * game.num_piece / valid_pieces[1])
            elif index_i == len(valid_pieces) - 1:
                schwarz_per_piece.append((game.board_size - 1 - min(state.board['R'][piece_i])) * game.num_piece / (game.num_piece - valid_pieces[-2] - 1))
            else:
                schwarz_per_piece.append((game.board_size - 1 - min(state.board['R'][piece_i])) * game.num_piece / (valid_pieces[index_i + 1] - valid_pieces[index_i - 1] - 1))
        schwarz['R'] = min(schwarz_per_piece)
    valid_pieces = [piece_i for piece_i in range(game.num_piece) if state.board['B'][piece_i] is not None]
    if len(valid_pieces) == 0:
        schwarz['B'] = (game.board_size - 1) * game.num_piece
    elif len(valid_pieces) == 1:
        schwarz['B'] = max(state.board['B'][valid_pieces[0]])
    else:
        schwarz_per_piece = []
        for index_i, piece_i in enumerate(valid_pieces):
            if index_i == 0:
                schwarz_per_piece.append(max(state.board['B'][piece_i]) * game.num_piece / valid_pieces[1])
            elif index_i == len(valid_pieces) - 1:
                schwarz_per_piece.append(max(state.board['B'][piece_i]) * game.num_piece / (game.num_piece - valid_pieces[-2] - 1))
            else:
                schwarz_per_piece.append(max(state.board['B'][piece_i]) * game.num_piece / (valid_pieces[index_i + 1] - valid_pieces[index_i - 1] - 1))
        schwarz['B'] = min(schwarz_per_piece)
    return schwarz

def schwarz_diff_to_weight(diff, max_schwarz):
    # Task 3
    # Return a weight value based on the relative difference in Schwarz scores.
    # Replace the line below with your code.
    
    # Normalize the difference
    normalized_diff = diff / max_schwarz
    # Define the intervals and corresponding weights as specified in Table 1
    if normalized_diff < -0.5:
        return 1
    elif -0.5 <= normalized_diff < -0.375:
        return 2
    elif -0.375 <= normalized_diff < -0.25:
        return 4
    elif -0.25 <= normalized_diff < -0.125:
        return 8
    elif -0.125 <= normalized_diff < 0:
        return 16
    elif 0 <= normalized_diff < 0.125:
        return 32
    elif 0.125 <= normalized_diff < 0.25:
        return 64
    elif 0.25 <= normalized_diff < 0.375:
        return 128
    elif 0.375 <= normalized_diff < 0.5:
        return 256
    elif normalized_diff >= 0.5:
        return 512

def random_policy(game, state):
    return random.choice(list(game.actions(state)))

def schwarz_policy(game, state):
    actions = list(game.actions(state))
    to_move = state.to_move
    opponent = 'B' if to_move == 'R' else 'R'
    weights = []
    for action in actions:
        state_prime = game.result(state, action)
        schwarz = schwarz_score(game, state_prime)
        schwarz_diff = schwarz[opponent] - schwarz[to_move]
        weights.append(schwarz_diff_to_weight(schwarz_diff, (game.board_size - 1) * game.num_piece))
    return random.choices(actions, weights=weights)[0]

def random_mcts_player(game, state):
    return stochastic_monte_carlo_tree_search(state, game, random_policy, 100)

def schwarz_mcts_player(game, state):
    return stochastic_monte_carlo_tree_search(state, game, schwarz_policy, 100)

if __name__ == '__main__':

    # Task 1 test code
    
    num_win = 0
    num_loss = 0
    for _ in range(50):
        if EinStein().play_game(alpha_beta_player, random_player) == 1:
            num_win += 1
        else:
            num_loss += 1
    for _ in range(50):
        if EinStein().play_game(random_player, alpha_beta_player) == 1:
            num_loss += 1
        else:
            num_win += 1
    print(f'alpha-beta pruned minimax player vs. random-move player: {num_win} wins and {num_loss} losses', end='\n\n')
    

    # Task 2 test code
    
    num_win = 0
    num_loss = 0
    for _ in range(50):
        if MehrSteine(4).play_game(random_mcts_player, random_player) == 1:
            num_win += 1
        else:
            num_loss += 1
    for _ in range(50):
        if MehrSteine(4).play_game(random_player, random_mcts_player) == 1:
            num_loss += 1
        else:
            num_win += 1
    print(f'MCTS with random playout vs. random-move player: {num_win} wins and {num_loss} losses', end='\n\n')
    
    
    # Task 3 test code
    
    num_win = 0
    num_loss = 0
    for _ in range(50):
        if MehrSteine(4).play_game(schwarz_mcts_player, random_mcts_player) == 1:
            num_win += 1
        else:
            num_loss += 1
    for _ in range(50):
        if MehrSteine(4).play_game(random_mcts_player, schwarz_mcts_player) == 1:
            num_loss += 1
        else:
            num_win += 1
    print(f'MCTS with Schwarz-based playout vs. MCTS with random playout: {num_win} wins and {num_loss} losses', end='\n\n')
