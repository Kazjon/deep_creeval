__author__ = 'kazjon'
import sys

class MCTSDesignSpace(object):
	def __init__(self, expectation_model, design_features, plausibility_dist, starting_features=(), min_moves=3, max_moves=12, end_token = "END"):
		self.model = expectation_model
		self.features = design_features
		self.starting_features = starting_features
		self.plausibility_dist = plausibility_dist
		self.end_token = end_token
		self.min_moves = min_moves
		self.max_moves = max_moves

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
		if len(state_history[-1]) == self.max_moves:
			return [self.end_token]
		if len(state_history[-1]) >= self.min_moves:
			return [self.end_token] + [f for f in self.features if f not in (self.starting_features + state_history[-1])]
		else:
			return [f for f in self.features if f not in (self.starting_features + state_history[-1])]

	def score(self, state_history):
		# Takes a sequence of game states representing the full
		# game history.  If the design is complete, return its score.
		# Otherwise return -1.
		if self.end_token in state_history[-1]:
			dict_state = {f:1 for f in state_history[-1]}
			ll = -self.model.recon_cost(dict_state)[0]
			score = 1 - min(1,(max(0,ll + self.plausibility_dist["max"]))/-self.plausibility_dist["min"])
			return score
		return -1