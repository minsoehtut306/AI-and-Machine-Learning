"""
EinStein würfelt nicht! (EWN) — Adversarial & Stochastic Search

Implements two variants of the EWN game:
- EinStein: 3x3 board, 1 piece per player (deterministic).
- MehrSteine: k x k board, n pieces per player with stochastic dice-roll turns.

Includes:
- Alpha-beta vs random benchmark (EinStein).
- Stochastic Monte Carlo Tree Search (MCTS) with:
  * Random playout policy
  * Schwarz-score-weighted playout policy

Requires AIMA's `games4e.py` (Game, StochasticGame, GameState, StochasticGameState, MCT_Node, etc.).
"""

import math
import random
from copy import deepcopy
from games4e import *  # AIMA "games4e.py"


# ---------------------------------------------------------------------------
# 3x3, one-piece-per-player, deterministic version
# ---------------------------------------------------------------------------

class EinStein(Game):
    """
    3x3 deterministic EWN:
    - Red starts at (0, 0), moves first; moves: south, east, southeast.
    - Blue starts at (2, 2); moves: north, west, northwest.
    - Capture by moving onto a tile occupied by any piece (own or opponent).
    - Win if you reach opponent corner or capture opponent's last piece.
    """

    def __init__(self):
        # Initial GameState: Red to move, utility=0 (non-terminal),
        # board with positions, and precomputed legal moves for Red.
        self.initial = GameState(
            to_move='R',
            utility=0,
            board={'R': (0, 0), 'B': (2, 2)},
            moves=[(1, 1), (1, 0), (0, 1)]
        )

    def compute_moves(self, board, to_move):
        """Compute legal moves for `to_move` on a 3x3 board, given positions."""
        moves = []
        if board[to_move]:
            r, c = board[to_move]
            if to_move == 'R':
                # Down
                if r < 2:
                    moves.append((r + 1, c))
                    # Down-right
                    if c < 2:
                        moves.append((r + 1, c + 1))
                # Right
                if c < 2:
                    moves.append((r, c + 1))
            else:  # 'B'
                # Up
                if r > 0:
                    moves.append((r - 1, c))
                    # Up-left
                    if c > 0:
                        moves.append((r - 1, c - 1))
                # Left
                if c > 0:
                    moves.append((r, c - 1))
        return moves

    def display(self, state):
        """Print a small board with 'R' and 'B' markers."""
        displayed_board = [[' ' for _ in range(3)] for _ in range(3)]
        for player_i in ['R', 'B']:
            if state.board[player_i] is not None:
                r, c = state.board[player_i]
                displayed_board[r][c] = f'{player_i}'
        print('\n'.join(['|' + '|'.join(row) + '|' for row in displayed_board]), end='\n\n')

    def terminal_test(self, state):
        """Terminal if stored utility != 0 (win/loss)"""
        return state.utility != 0

    def actions(self, state):
        """Legal moves are precomputed and stored in the state."""
        return state.moves

    # --- Core game logic ---

    def result(self, state, move):
        """
        Apply a move and return the next GameState.
        - Move the current player's piece to `move`.
        - Capture if destination equals opponent's position.
        - Switch turn; recompute utility and opponent's moves.
        """
        player = state.to_move
        opponent = 'B' if player == 'R' else 'R'

        board_copy = deepcopy(state.board)
        # Move current player's piece
        board_copy[player] = move

        # Capture if occupying opponent's position
        if board_copy[opponent] == move:
            board_copy[opponent] = None

        # Compute new stored utility (to Red) and legal moves for next player
        new_utility = self.compute_utility(board_copy)
        new_moves = self.compute_moves(board_copy, opponent)

        return GameState(
            to_move=opponent,
            utility=new_utility,
            board=board_copy,
            moves=new_moves
        )

    def utility(self, state, player):
        """
        Utility is stored as value to Red.
        Return utility(state) to `player` (negate if Blue).
        """
        return state.utility if player == 'R' else -state.utility

    def compute_utility(self, board):
        """
        Compute board utility to Red:
        - Red loses (−1): Red piece removed OR Blue reaches (0, 0)
        - Red wins (+1): Blue piece removed OR Red reaches (2, 2)
        - Else 0 (non-terminal)
        """
        if board['R'] is None or board['B'] == (0, 0):
            return -1
        if board['B'] is None or board['R'] == (2, 2):
            return 1
        return 0


# ---------------------------------------------------------------------------
# k x k, multiple pieces per player, stochastic version
# ---------------------------------------------------------------------------

class MehrSteine(StochasticGame):
    """
    Generalized EWN on k x k board with n pieces per player.
    - Piece to move is chosen by an n-sided die (uniform).
    - If chosen piece is gone, nearest lower or higher index is selected.
    - Movement rules as in EWN (toward opponent corner, 8-neighbour reduced).
    - Capture by moving onto occupied tile.
    """

    def __init__(self, board_size):
        assert board_size >= 3
        self.board_size = board_size
        self.num_piece = int((board_size - 1) * (board_size - 2) / 2)

        # Build starting corner triangles
        board = {'R': [], 'B': []}
        for i in range(board_size - 2):
            for j in range(board_size - 2 - i):
                board['R'].append((i, j))
                board['B'].append((board_size - 1 - i, board_size - 1 - j))

        # Start with no precomputed moves/chance; MCTS will fill these in
        self.initial = StochasticGameState(
            to_move='R', utility=0, board=board, moves=None, chance=None
        )

    def compute_moves(self, board, to_move, index):
        """
        For a given piece index and player, return legal destination moves:
        Red: south, east, southeast;  Blue: north, west, northwest
        """
        moves = []
        coordinates = board[to_move][index]
        r, c = coordinates

        if to_move == 'R':
            if r < self.board_size - 1:
                moves.append((index, (r + 1, c)))
                if c < self.board_size - 1:
                    moves.append((index, (r + 1, c + 1)))
            if c < self.board_size - 1:
                moves.append((index, (r, c + 1)))
        else:  # 'B'
            if r > 0:
                moves.append((index, (r - 1, c)))
                if c > 0:
                    moves.append((index, (r - 1, c - 1)))
            if c > 0:
                moves.append((index, (r, c - 1)))

        return moves

    def display(self, state):
        """Pretty-print board with piece indices (R#, B#)."""
        spacing = 1 if self.num_piece == 1 else math.floor(math.log(self.num_piece - 1, 10)) + 1
        displayed_board = [[' ' * (spacing + 1) for _ in range(self.board_size)]
                           for _ in range(self.board_size)]
        for player_i in ['R', 'B']:
            for piece_i in range(self.num_piece):
                pos = state.board[player_i][piece_i]
                if pos is not None:
                    r, c = pos
                    displayed_board[r][c] = player_i + str(piece_i).rjust(spacing)
        print('\n'.join(['|' + '|'.join(row) + '|' for row in displayed_board]), end='\n\n')

    def terminal_test(self, state):
        """Terminal if stored utility != 0 (win/loss)."""
        return state.utility != 0

    def actions(self, state):
        """Legal moves are stored in state (after chance outcome)."""
        return state.moves

    # --- Core game logic ---

    def result(self, state, move):
        """
        Apply a move (index, (r, c)) and return next StochasticGameState.
        - Move selected piece; capture any opponent piece on that tile.
        - Switch turn; recompute utility.
        - Do NOT compute next moves/chance yet (set to None).
        """
        board_copy = deepcopy(state.board)
        current_player = state.to_move
        opponent = 'R' if current_player == 'B' else 'B'

        idx, destination = move
        board_copy[current_player][idx] = destination

        # Capture any opponent piece at destination
        for oi, opos in enumerate(board_copy[opponent]):
            if opos == destination:
                board_copy[opponent][oi] = None

        new_utility = self.compute_utility(board_copy)
        return StochasticGameState(
            to_move=opponent,
            utility=new_utility,
            board=board_copy,
            moves=None,
            chance=None
        )

    def utility(self, state, player):
        """Return stored utility to `player` (stored as Red's utility)."""
        return state.utility if player == 'R' else -state.utility

    def compute_utility(self, board):
        """
        Compute board utility to Red:
        - Red loses (−1): all Red None OR any Blue at (0,0)
        - Red wins (+1): all Blue None OR any Red at (k-1, k-1)
        - Else 0
        """
        top_left = (0, 0)
        bottom_right = (self.board_size - 1, self.board_size - 1)

        red_all_gone = all(pos is None for pos in board['R'])
        blue_all_gone = all(pos is None for pos in board['B'])
        blue_reached_red_corner = any(pos == top_left for pos in board['B'] if pos is not None)
        red_reached_blue_corner = any(pos == bottom_right for pos in board['R'] if pos is not None)

        if red_all_gone or blue_reached_red_corner:
            return -1
        if blue_all_gone or red_reached_blue_corner:
            return 1
        return 0

    # --- Stochastic (chance) interface for MCTS ---

    def chances(self, state):
        """
        Possible dice-roll outcomes: 0..(n-1), where n is starting piece count.
        Uniform distribution assumed.
        """
        return list(range(self.num_piece))

    def outcome(self, state, chance):
        """
        Given a dice outcome `chance` (piece index), return a state with legal
        moves computed for the piece to move. If that piece is gone, select the
        nearest lower/higher index that still exists. If none exist, no moves.
        """
        to_move = state.to_move
        pieces = state.board[to_move]

        # Find actual piece index to move (nearest available index if chosen is gone)
        chosen = chance
        if pieces[chosen] is None:
            # Search nearest lower, then higher
            lower = list(range(chosen - 1, -1, -1))
            higher = list(range(chosen + 1, len(pieces)))
            found = None
            for idx in lower + higher:
                if pieces[idx] is not None:
                    found = idx
                    break
            chosen = found if found is not None else None

        if chosen is None:
            legal_moves = []
        else:
            legal_moves = self.compute_moves(state.board, to_move, chosen)

        return StochasticGameState(
            to_move=state.to_move,
            utility=state.utility,
            board=state.board,
            moves=legal_moves,
            chance=chosen
        )

    def probability(self, chance):
        """Uniform probability over n starting pieces."""
        return 1 / self.num_piece


# ---------------------------------------------------------------------------
# Monte Carlo Tree Search (stochastic) with playout policies
# ---------------------------------------------------------------------------

def stochastic_monte_carlo_tree_search(state, game, playout_policy, N=1000):
    """
    Stochastic MCTS:
    - Selection/Expansion use UCB over action+chance branches.
    - Simulation uses given `playout_policy`.
    - Backprop toggles utility sign up the tree (two-player zero-sum).
    """

    def select(n):
        # Descend by UCB until leaf
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        # Expand leaf: enumerate action + chance children
        if not n.children and not game.terminal_test(n.state):
            n.children = {
                MCT_Node(state=game.outcome(game.result(n.state, action), chance), parent=n): action
                for action in game.actions(n.state)
                for chance in game.chances(game.result(n.state, action))
            }
        return select(n)

    def simulate(game, state):
        # Play out randomly / with policy until terminal; return utility to root player
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = playout_policy(game, state)
            state = game.result(state, action)
            chance = random.choice(game.chances(state))
            state = game.outcome(state, chance)
        v = game.utility(state, player)
        return -v  # backprop step expects negation pattern

    def backprop(n, utility):
        # Add positive utility; propagate alternating signs up to root
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

    # Choose most visited child
    max_state = max(root.children, key=lambda p: p.N)
    return root.children.get(max_state)


# ---------------------------------------------------------------------------
# Schwarz scoring & weighted playout policy
# ---------------------------------------------------------------------------

def schwarz_score(game, state):
    """
    Heuristic 'Schwarz' score: estimated turns to goal (assuming no captures).
    Lower is better. Returns dict {'R': score, 'B': score}.
    """
    schwarz = {}

    # Red
    valid = [i for i in range(game.num_piece) if state.board['R'][i] is not None]
    if len(valid) == 0:
        schwarz['R'] = (game.board_size - 1) * game.num_piece
    elif len(valid) == 1:
        schwarz['R'] = game.board_size - 1 - min(state.board['R'][valid[0]])
    else:
        per = []
        for idx, pidx in enumerate(valid):
            if idx == 0:
                per.append((game.board_size - 1 - min(state.board['R'][pidx])) * game.num_piece / valid[1])
            elif idx == len(valid) - 1:
                per.append((game.board_size - 1 - min(state.board['R'][pidx])) * game.num_piece /
                           (game.num_piece - valid[-2] - 1))
            else:
                per.append((game.board_size - 1 - min(state.board['R'][pidx])) * game.num_piece /
                           (valid[idx + 1] - valid[idx - 1] - 1))
        schwarz['R'] = min(per)

    # Blue
    valid = [i for i in range(game.num_piece) if state.board['B'][i] is not None]
    if len(valid) == 0:
        schwarz['B'] = (game.board_size - 1) * game.num_piece
    elif len(valid) == 1:
        schwarz['B'] = max(state.board['B'][valid[0]])
    else:
        per = []
        for idx, pidx in enumerate(valid):
            if idx == 0:
                per.append(max(state.board['B'][pidx]) * game.num_piece / valid[1])
            elif idx == len(valid) - 1:
                per.append(max(state.board['B'][pidx]) * game.num_piece /
                           (game.num_piece - valid[-2] - 1))
            else:
                per.append(max(state.board['B'][pidx]) * game.num_piece /
                           (valid[idx + 1] - valid[idx - 1] - 1))
        schwarz['B'] = min(per)

    return schwarz


def schwarz_diff_to_weight(diff, max_schwarz):
    """
    Map relative Schwarz-score difference to a categorical weight (Table mapping).

    diff = schwarz[opponent] - schwarz[to_move]
    normalized = diff / max_schwarz

    Intervals → Weights:
      (-inf, -0.5)  :   1
      [-0.5,-0.375) :   2
      [-0.375,-0.25):   4
      [-0.25,-0.125):   8
      [-0.125,0)    :  16
      [0,0.125)     :  32
      [0.125,0.25)  :  64
      [0.25,0.375)  : 128
      [0.375,0.5)   : 256
      [0.5, inf)    : 512
    """
    normalized = diff / max_schwarz if max_schwarz != 0 else 0.0
    if normalized < -0.5:
        return 1
    elif -0.5 <= normalized < -0.375:
        return 2
    elif -0.375 <= normalized < -0.25:
        return 4
    elif -0.25 <= normalized < -0.125:
        return 8
    elif -0.125 <= normalized < 0:
        return 16
    elif 0 <= normalized < 0.125:
        return 32
    elif 0.125 <= normalized < 0.25:
        return 64
    elif 0.25 <= normalized < 0.375:
        return 128
    elif 0.375 <= normalized < 0.5:
        return 256
    else:  # normalized >= 0.5
        return 512


# ---------------------------------------------------------------------------
# Playout policies & wrappers
# ---------------------------------------------------------------------------

def random_policy(game, state):
    """Uniformly random action from legal actions."""
    return random.choice(list(game.actions(state)))

def schwarz_policy(game, state):
    """
    Weighted playout policy based on Schwarz-score differences.
    Higher weight if the action improves relative Schwarz score.
    """
    actions = list(game.actions(state))
    to_move = state.to_move
    opponent = 'B' if to_move == 'R' else 'R'
    weights = []

    for action in actions:
        # One-step lookahead via result+chance (chance weighted later by MCTS)
        state_prime = game.result(state, action)
        schwarz = schwarz_score(game, state_prime)
        schwarz_diff = schwarz[opponent] - schwarz[to_move]
        weights.append(schwarz_diff_to_weight(schwarz_diff, (game.board_size - 1) * game.num_piece))

    return random.choices(actions, weights=weights)[0]


def random_mcts_player(game, state):
    return stochastic_monte_carlo_tree_search(state, game, random_policy, N=100)

def schwarz_mcts_player(game, state):
    return stochastic_monte_carlo_tree_search(state, game, schwarz_policy, N=100)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # --- Deterministic EinStein: alpha-beta vs random ---
    num_win, num_loss = 0, 0
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
    print(f'Alpha-beta minimax vs random (EinStein 3x3): {num_win} wins / {num_loss} losses\n')

    # --- Stochastic MehrSteine (k=4): MCTS (random playout) vs random ---
    num_win, num_loss = 0, 0
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
    print(f'MCTS (random playout) vs random (MehrSteine 4x4): {num_win} wins / {num_loss} losses\n')

    # --- Stochastic MehrSteine (k=4): MCTS (Schwarz playout) vs MCTS (random) ---
    num_win, num_loss = 0, 0
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
    print(f'MCTS (Schwarz playout) vs MCTS (random) (MehrSteine 4x4): {num_win} wins / {num_loss} losses\n')
