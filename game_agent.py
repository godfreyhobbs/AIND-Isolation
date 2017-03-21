"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
from __future__ import print_function
import logging

logging.basicConfig(level=logging.ERROR)

import numpy as np

TIME_CUT_OFF = 1

TEST_MODE = True

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player, turns=0):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    moves = game.get_legal_moves(player)
    own_moves = len(moves)

    turns = (game.width * game.height - len(game.get_blank_spaces())) * .5
    return float(own_moves + turns                 )


#
# def custom_score(game, player, turns=0):
#     """Calculate the heuristic value of a game state from the point of view
#     of the given player.
#
#     Note: this function should be called from within a Player instance as
#     `self.score()` -- you should not need to call this function directly.
#
#     Parameters
#     ----------
#     game : `isolation.Board`
#         An instance of `isolation.Board` encoding the current state of the
#         game (e.g., player locations and blocked cells).
#
#     player : object
#         A player instance in the current game (i.e., an object corresponding to
#         one of the player objects `game.__player_1__` or `game.__player_2__`.)
#
#     Returns
#     -------
#     float
#         The heuristic value of the current game state to the specified player.
#     """
#     # The score of a specified game state is just the number of moves open to the active player
#
#     # OpenMoveEvalFn()
#     if game.is_loser(player):
#         return float("-inf")
#
#     if game.is_winner(player):
#         return float("inf")
#
#     moves = game.get_legal_moves(player)
#
#     own_moves = len(moves)
#     opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
#
#     edge_penalty = sum([(x - 3) * (x - 3) + (y - 3) * (y - 3) for x, y in moves]) / ((3 * 3 * 2 * len(moves)) + .001)
#     return float(own_moves - opp_moves) - edge_penalty

# def custom_score(game, player, turns=0):
#         """Calculate the heuristic value of a game state from the point of view
#         of the given player.
#
#         Note: this function should be called from within a Player instance as
#         `self.score()` -- you should not need to call this function directly.
#
#         Parameters
#         ----------
#         game : `isolation.Board`
#             An instance of `isolation.Board` encoding the current state of the
#             game (e.g., player locations and blocked cells).
#
#         player : object
#             A player instance in the current game (i.e., an object corresponding to
#             one of the player objects `game.__player_1__` or `game.__player_2__`.)
#
#         Returns
#         -------
#         float
#             The heuristic value of the current game state to the specified player.
#         """
#         if game.is_loser(player):
#             return float("-inf")
#
#         if game.is_winner(player):
#             return float("inf")
#
#         moves = game.get_legal_moves(player)
#         # try to avoid the corners
#         no_corners = [move for move in moves if move != (0, 0) and move != (0, 7) and move != (7, 0) and move != (7, 7)]
#         no_corner_bonus = len(no_corners)
#
#         own_moves = len(moves)
#         opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
#
#         return float(own_moves + no_corner_bonus - opp_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.TIMER_THRESHOLD = timeout
        self.eval_fn = score_fn

        # keep table of [moves] => (largest_depth, max_move,  max_value) then use for lookup/memozation.
        self.nextMaxMove = {}
        # consider keeping a new parallel next move table for opponent
        # keep table of [moves] => (largest_depth, min_move, min_value) then use for lookup/memozation.
        self.nextMinMove = {}
        self.first_mover = False
        self.turns = 0
        self.last_explorered_move_and_forecast = ((-1, -1), None)
        self.last_player_move = (-1, -1)
        self.indexArray = None
        # {game_state: {depth: {move: (val, child_move)}}}
        self.getMax_MiniMax_cache = {}
        # {game_state: {depth: {move: (val, child_move)}}}
        self.getMin_MiniMax_cache = {}

    # for  symmetry
    def init_indicee(self, game):
        self.indexArray = [[(j, i) for i in range(game.width)] for j in range(game.height)]

        ###indexes
        self.flipud_index = np.flipud(self.indexArray)
        self.fliplr_index = np.fliplr(self.indexArray)
        self.fliplr_flipud_index = np.fliplr(self.flipud_index)
        self.rot_90_index = np.rot90(self.indexArray)
        self.rot_180_index = np.rot90(self.indexArray, 2)
        self.flipud_rot_90_index = np.flipud(self.rot_90_index)
        self.flipud_rot_180_index = np.flipud(self.rot_180_index)
        self.indicee = (self.rot_180_index, self.flipud_index, self.flipud_rot_180_index, self.fliplr_index,
                        self.fliplr_flipud_index, self.rot_90_index,
                        self.flipud_rot_90_index, self.flipud_rot_180_index)

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        logging.debug(
            'starting getmove - method {} depth {} with threshold_time {} iter {} timeleft {} legal_moves {}'.format(
                self.method,
                self.search_depth,
                self.TIMER_THRESHOLD,
                self.iterative,
                time_left(),
                legal_moves))
        self.time_left = time_left

        if self.indexArray == None:
            self.init_indicee(game)
        # returning immediately if there are no legal moves
        if len(legal_moves) == 0:
            return None, None
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or
        DUMMY_MOVE = (42, 42)
        move_to_return = DUMMY_MOVE

        if len(legal_moves) is 0:
            return move_to_return

        if len(legal_moves) == game.width * game.height:
            self.first_mover = True
            self.turns = 0

        if len(legal_moves) == game.width * game.height - 1:
            self.turns = 0
            self.first_mover = False

        self.turns = self.turns + 1

        # test fail so commented out
        # if moving take center whenever possible
        GRAB_CENTER_MOVE = not TEST_MODE
        if GRAB_CENTER_MOVE and (int(game.width / 2), int(game.height / 2)) in legal_moves:
            # no need to memorize
            return (int(game.width / 2), int(game.height / 2))

        # print 'first mover [',self.first_mover,'][',self.turns

        # test fail so commented out
        # if you are player one try reflection:
        USE_REFLECT_MOVE = not TEST_MODE

        if USE_REFLECT_MOVE and move_to_return == DUMMY_MOVE:
            for index_matrix in self.indicee:
                if self.first_mover:
                    reflect_move = self.get_reflect_move(game, index_matrix, legal_moves)
                    if [x for x in legal_moves if x[0] == reflect_move[0] and x[1] == reflect_move[1]]:
                        if move_to_return == DUMMY_MOVE:
                            return reflect_move

        curr_depth = 1
        result_val = None
        result_move = None
        try:

            while (self.iterative or self.search_depth < curr_depth) and not (result_val and np.isinf(result_val)):
                if self.method == 'minimax':
                    result_val, result_move = self.minimax(game, depth=curr_depth)
                else:
                    result_val, result_move = self.alphabeta(game, depth=curr_depth)

                # if curr_depth > 3:
                #     print "move function  [",curr_depth ,"][",time_left() ,"][",best_move,"value",best_val,""
                move_to_return = result_move
                curr_depth = curr_depth + 1

            # print "move function returning [",curr_depth ,"][",time_left() ,"][",best_move,"value",best_val,
            if move_to_return == DUMMY_MOVE:
                move_to_return = legal_moves[0]

        except Timeout:
            # Handle any actions required at timeout, if necessary
            # check for timeout
            logging.debug(
                'getmove Timeout - method {} curr depth {} with threshold_time {} iter {} timeleft {} val {} move {}'.format(
                    self.method,
                    curr_depth,
                    self.TIMER_THRESHOLD,
                    self.iterative,
                    time_left(),
                    result_val,
                    result_move))

            if result_move is None:
                # print "move function returning early  [",curr_depth ,"][",time_left() ,"][",best_move,"value",best_val,
                if move_to_return == DUMMY_MOVE:
                    move_to_return = legal_moves[0]

        self.last_player_move = move_to_return
        # Return the best move from the last completed search iteration
        return move_to_return

    def get_reflect_move(self, game, index_matrix, legal_moves):
        # HACK TRYING OUT get_opponent
        # other = game.forecast_move(legal_moves[0]).get_active_player()
        other = game.forecast_move(legal_moves[0]).get_opponent(self)
        lastM = game.get_last_move_for_player(other)
        reflect_move = index_matrix[lastM[0]][lastM[1]]
        reflect_move = (reflect_move[0], reflect_move[1])
        return reflect_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        max_min_msg = 'MAX' if maximizing_player else 'MIN'
        turn_count = self.turns + self.search_depth - depth
        logging.debug('starting alphabeta as {} depth {} with time {} alpha {} beta {}'.format(max_min_msg, depth,
                                                                                               self.time_left(), alpha,
                                                                                               beta))
        # logging.debug('\n{}'.format(game.to_string()))

        if self.indexArray == None:
            self.init_indicee(game)

        if self.time_left() < self.TIMER_THRESHOLD:
            logging.debug('alphabeta raising timeout {} threshold {}'.format(self.time_left(), self.TIMER_THRESHOLD))
            raise Timeout()

        get_state = game.__board_state__
        best_move, best_val = self.checkforTerminalStates(game)

        if best_val is not None:
            return best_val, best_move

        # need to find the best move then forecast
        # if too deep return score but do not save
        # go deeper
        if maximizing_player:
            best_val, best_move = self.getMax_Alpha_Beta(game, depth, alpha, beta, move=None)
        else:
            best_val, best_move = self.getMin_Alpha_Beta(game, depth, alpha, beta, move=None)

        if best_move is None:
            for move in game.get_legal_moves():
                if best_val == self.utility(game.forecast_move(move), self, move=move):
                    best_move = move
                    break

        logging.debug(
            'returning {} move {} value {} depth {} alpha {} beta {}'.format(max_min_msg, best_move, best_val, depth,
                                                                             alpha, beta))
        return best_val, best_move

    def getMax_Alpha_Beta(self, game, depth, alpha, beta, move):
        # TODO: check for terminal states - win lose
        if depth == 0:
            return self.utility(game, self, move=move), move

        best_val = float("-inf")
        best_move = None
        for move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            curr_val, curr_move = self.getMin_Alpha_Beta(game.forecast_move(move), depth - 1, alpha, beta, move)

            if curr_val > best_val:
                best_val = curr_val
                best_move = move

            if best_val >= beta:
                return best_val, move

            alpha = max(alpha, curr_val)

        return best_val, best_move

    def getMin_Alpha_Beta(self, game, depth, alpha, beta, move):
        # TODO: check for terminal states e.g win or lose
        if depth == 0:
            return self.utility(game, self, move), move

        best_val = float("inf")
        best_move = None
        for move in game.get_legal_moves():
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            curr_val, curr_move = self.getMax_Alpha_Beta(game.forecast_move(move), depth - 1, alpha, beta, move)

            if curr_val < best_val:
                best_val = curr_val
                best_move = move

            if best_val <= alpha:
                return best_val, move

            beta = min(beta, best_val)

        return best_val, best_move

    def checkforTerminalStates(self, game):
        get_state = game.__board_state__
        best_move = (42, 42)
        best_val = None
        # Check terminal states
        if game.is_winner(self):
            best_move, best_val = best_move, float("inf")
        if game.is_loser(self):
            best_move, best_val = best_move, float("-inf")
        if str(game.__board_state__) in self.nextMaxMove and np.isinf(self.nextMaxMove[str(get_state)][2]):
            best_move, best_val = self.nextMaxMove[str(get_state)][1], self.nextMaxMove[str(get_state)][2]
        return best_move, best_val

    # value for a give game state and a givenplayer
    def utility(self, game, player, move=None, turn_count=None):
        # # Check terminal states
        # if game.is_winner(self):
        #     return float("inf")
        #
        # if game.is_loser(self):
        #     return float("-inf")
        #
        # result = self.score(game, player)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        result = self.score(game, player)
        logging.debug('utility for move [{}] returning [{}]'.format(move, result))

        return result


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        max_min_msg = 'MAX' if maximizing_player else 'MIN'
        turn_count = self.turns + self.search_depth - depth
        logging.debug('starting minmax as {} depth {} with time {}'.format(max_min_msg, depth, self.time_left()))
        # logging.debug('\n{}'.format(game.to_string()))

        if self.indexArray == None:
            self.init_indicee(game)

        if self.time_left() < self.TIMER_THRESHOLD:
            logging.debug('minimax raising timeout {} threshold {}'.format(self.time_left(), self.TIMER_THRESHOLD))
            raise Timeout()

        get_state = game.__board_state__
        best_move, best_val = self.checkforTerminalStates(game)

        if best_val is not None:
            return best_val, best_move

        # need to find the best move then forecast
        # if too deep return score but do not save
        # go deeper
        if maximizing_player:
            best_val, best_move = self.getMax_MiniMax(game, depth, move=None)
        else:
            best_val, best_move = self.getMin_MiniMax(game, depth, move=None)

        if best_move is None:
            for move in game.get_legal_moves():
                if best_val == self.utility(game.forecast_move(move), self, move=move):
                    best_move = move
                    break
        logging.debug('returning minimax {} move {} value {} depth {}'.format(max_min_msg, best_move, best_val, depth))
        return best_val, best_move

    def getMax_MiniMax(self, game, depth, move):

        # {game_state: {depth: {move: (val, child_move)}}}
        game_state_str = str(game.__board_state__)
        cache = self.getMax_MiniMax_cache

        if not TEST_MODE and game_state_str in cache:
            if depth in cache[game_state_str]:
                raise 'getMax_MiniMax found cache hit 1'
                if move in cache[game_state_str][depth]:
                    raise 'getMax_MiniMax found cache hit'
                    logging.debug('getMax_MiniMax returning cached move {}'.format((game_state_str,depth,move)))
                    return cache[game_state_str][depth][move]


        # TODO: check for terminal states - win lose
        if depth == 0:
            utility = self.utility(game, self, move)
            # {game_state: {depth: {move: (val, child_move)}}}
            cached_value = {depth: {move: (utility, move)}}
            logging.debug('getMax_MiniMax cached value {}'.format(cached_value))
            cache[game_state_str] = cached_value
            return utility, move

        best_val = float("-inf")
        best_move = None
        legal_moves = game.get_legal_moves()
        logging.debug('getMax_MiniMax game {} depth {} move {} legalmoves {}'.format(game,depth,move,legal_moves))

        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            curr_val, curr_move = self.getMin_MiniMax(game.forecast_move(move), depth - 1, move)

            if curr_val > best_val:
                best_val = curr_val
                best_move = move

        # {game_state: {depth: {move: (val, child_move)}}}
        cached_value = {depth: {move: (best_val, best_move)}}
        logging.debug('getMax_MiniMax cached value {}'.format(cached_value))
        cache[game_state_str] = cached_value
        return best_val, best_move

    def getMin_MiniMax(self, game, depth, move):
        # {game_state: {depth: {move: (val, child_move)}}}
        game_state_str = str(game.__board_state__)
        cache = self.getMin_MiniMax_cache
        if  not TEST_MODE and game_state_str in cache:
            if depth in cache[game_state_str]:
                # raise 'getMin_MiniMax found cache hit 1'
                if move in cache[game_state_str][depth]:
                    # raise 'getMin_MiniMax found cache hit'
                    logging.debug('getMin_MiniMax returning cached move {}'.format((game_state_str,depth,move)))
                    return cache[game_state_str][depth][move]

        # TODO: check for terminal states e.g win or lose
        if depth == 0:
            utility = self.utility(game, self, move)
            # {game_state: {depth: {move: (val, child_move)}}}
            cached_value = {depth: {move: (utility, move)}}
            logging.debug('getMin_MiniMax cached value {}'.format(cached_value))
            cache[game_state_str] = cached_value
            return utility, move

        best_val = float("inf")
        best_move = None
        legal_moves = game.get_legal_moves()
        logging.debug('getMin_MiniMax game {} depth {} move {} legalmoves {}'.format(game, depth, move, legal_moves))

        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            curr_val, curr_move = self.getMax_MiniMax(game.forecast_move(move), depth - 1, move)

            if curr_val < best_val:
                best_val = curr_val
                best_move = move

        # {game_state: {depth: {move: (val, child_move)}}}
        cached_value = {depth: {move: (best_val, best_move)}}
        logging.debug('getMin_MiniMax  cached value {}'.format(cached_value))
        cache[game_state_str] = cached_value

        return best_val, best_move

    def memoize(self, best_move, best_val, depth, game_state, move_cache):

        if self.turns > 3:
            return

        flipud_game_state = np.flipud(game_state)
        fliplr_game_state = np.fliplr(game_state)

        best_fliplr_move = self.fliplr_index[best_move[0]][best_move[1]]
        best_flipud_move = self.flipud_index[best_move[0]][best_move[1]]
        best_fliplr_flipud_move = self.fliplr_flipud_index[best_move[0]][best_move[1]]
        best_rot_90_move = self.rot_90_index[best_move[0]][best_move[1]]
        best_rot_180_move = self.rot_180_index[best_move[0]][best_move[1]]
        best_flipud_rot_90_move = self.flipud_rot_90_index[best_move[0]][best_move[1]]
        best_flipud_rot_180_move = self.flipud_rot_180_index[best_move[0]][best_move[1]]

        rot_90 = np.rot90(game_state)
        rot_180 = np.rot90(game_state, 2)
        # memoize game tree by printing (print board ) as the key

        move_cache[str(game_state)] = (depth, best_move, best_val)
        # consider not bothering with symetry if number moves is
        # less that 25 - 6 after 3 turns

        move_cache[str(fliplr_game_state)] = (depth, best_fliplr_move, best_val)
        move_cache[str(flipud_game_state)] = (depth, best_flipud_move, best_val)

        # if self is not game.get_active_player():
        move_cache[str(np.fliplr(flipud_game_state))] = (depth, best_fliplr_flipud_move, best_val)
        # move_cache[str(np.flipud(fliplr_game_state))] = (depth, best_move, best_val)
        move_cache[str(rot_90)] = (depth, best_rot_90_move, best_val)
        move_cache[str(rot_180)] = (depth, best_rot_180_move, best_val)
        move_cache[str(np.flipud(rot_90))] = (depth, best_flipud_rot_90_move, best_val)
        move_cache[str(np.flipud(rot_180))] = (depth, best_flipud_rot_180_move, best_val)

from isolation import Board
from sample_players import RandomPlayer
from sample_players import null_score
if __name__ == "__main__":
    counter = 0
    # (self, search_depth = 3, score_fn = custom_score,
    #                                     iterative = True, method = 'minimax', timeout = 10.)

    customPlayer = CustomPlayer(3, null_score)
    randomPlayer = RandomPlayer()

    game = Board(customPlayer, 3,3)
    # player_1 = CustomPlayer(2)
    num_trials = orginal_num_trials = 5
    while num_trials > 0:
        num_trials = num_trials - 1
        while True:
            game = Board(customPlayer, randomPlayer)
            game.play()