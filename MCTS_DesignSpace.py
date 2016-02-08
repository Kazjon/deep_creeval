__author__ = 'kazjon'
import sys

class MCTSDesignSpace(object):
	def __init__(self, expectation_model, design_features, win_threshold=-20, starting_features=(), min_moves=3, end_token = "END"):
		self.model = expectation_model
		self.features = design_features
		self.starting_features = starting_features
		self.win_threshold = win_threshold
		self.end_token = end_token
		self.min_moves = min_moves

	def start(self):
		# Returns a representation of the starting state of the game.
		return tuple(self.starting_features)

	def current_player(self, state):
		# Takes the game state and returns the current player's
		# number.
		#NOTE: We're going to get rid of this as designing is always single player (for us).
		return 1

	def next_state(self, state, play):
		# Takes the game state, and the move to be applied.
		# Returns the new game state.
		if type(state) is not tuple:
			state = (state,)
		return tuple(list(state)+[play]) #Game states need to be hashable, thus the mess

	def legal_plays(self, state_history):
		# Takes a sequence of game states representing the full
		# game history, and returns the full list of moves that
		# are legal plays for the current player.
		if len(state_history[-1]) >= self.min_moves:
			return [self.end_token] + [f for f in self.features if f not in (self.starting_features + state_history[-1])]
		else:
			return [f for f in self.features if f not in (self.starting_features + state_history[-1])]

	def winner(self, state_history):
		# Takes a sequence of game states representing the full
		# game history.  If the game is now won, return the player
		# number.  If the game is still ongoing, return zero.  If
		# the game is tied, return a different distinct value, e.g. -1.
		if self.end_token in state_history[-1]:
			dict_state = {f:1 for f in state_history[-1]}
			nll = self.model.recon_cost(dict_state)[0]
			if nll > self.win_threshold:
				return 1
			return -1
		return 0