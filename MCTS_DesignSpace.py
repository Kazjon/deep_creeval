__author__ = 'kazjon'
import sys
import numpy as np

class MCTSDesignSpace(object):
	def __init__(self, expectation_model, design_features, plausibility_distribution=None, length_distribution=None, surprise_distribution=None, errors_by_length=None, score_method = "plausibility", starting_features=(), min_moves=3, max_moves=12, end_token = "END", surprise_depth=1):
		self.model = expectation_model
		self.features = design_features
		self.starting_features = starting_features
		self.plausibility_dist = plausibility_distribution
		self.surprise_dist = surprise_distribution
		self.length_dist = length_distribution
		self.errors_by_length = errors_by_length
		self.end_token = end_token
		self.min_moves = min_moves
		self.max_moves = max_moves
		self.score_method = score_method
		self.surprise_depth = surprise_depth

	def start(self):
		# Returns a representation of the starting state of the game.
		return frozenset(self.starting_features)

	def current_player(self, state):
		# Takes the game state and returns the current player's
		# number.
		#NOTE: We're going to get rid of this as designing is always single player (for us).
		return 1

	def next_state(self, state, play):
		# Takes the game state, and the move to be applied.
		# Returns the new game state.
		if type(state) is not frozenset:
			state = frozenset(state)
		return frozenset(list(state)+[play]) #Game states need to be hashable, thus the mess

	def legal_plays(self, state_history):
		# Takes a sequence of game states representing the full
		# game history, and returns the full list of moves that
		# are legal plays for the current player.
		if len(state_history[-1]) == self.max_moves:
			return frozenset([self.end_token])
		if len(state_history[-1]) >= self.min_moves:
			return frozenset([self.end_token] + [f for f in self.features if f not in self.starting_features and f not in state_history[-1]])
		else:
			return frozenset([f for f in self.features if f not in self.starting_features and f not in state_history[-1]])

	def play_dist(self, state_history, legal_plays):
		play_dist = {}
		uniform_prob = 1.0/len(legal_plays)
		if self.end_token in legal_plays and self.length_dist is not None:
			length_prob = self.length_dist[len(state_history[-1])]
			play_dist[self.end_token] = length_prob
			uniform_prob = (1.0-length_prob)/len(legal_plays)
		for play in legal_plays:
			if play is not self.end_token:
				play_dist[play] = uniform_prob
		return play_dist

	def score(self, state_history):
		# Takes a sequence of game states representing the full
		# game history.  If the design is complete, return its score.
		# Otherwise return -1.
		if self.end_token in state_history[-1]:
			if self.score_method == "plausibility":
				return self.plausibility_score(state_history[-1])
			if self.score_method == "surprise":
				return self.surprise_score(state_history[-1])
			if self.score_method == "plausibility+surprise":
				return (self.plausibility_score(state_history[-1]) + self.surprise_score(state_history[-1]))/2.0
		return -1

	def plausibility_score(self, state, weight_by_length=True):
		dict_state = {f:1 for f in state}
		ll = -self.model.recon_cost(dict_state)[0]
		score = 1 - min(1,(max(0,ll + self.plausibility_dist["max"]))/-self.plausibility_dist["min"])
		if weight_by_length:
			score *= np.exp(self.plausibility_dist["mean"]) / np.exp(self.errors_by_length[len(dict_state.keys())-1]) #Subtract one because of the END token
		return score

	def surprise_score(self, state):
		state = set(state)
		state.remove(self.end_token)
		surprises = self.model.surprising_sets(list(state),depth_limit=self.surprise_depth,n_iter=10)
		surp_list = self.model.sorted_surprise(surprises)
		max_surp = self.model.max_surprise(surp_list)
		avg_surp = self.model.avg_surprise(surp_list)
		max_surp = min(1,max(0,max_surp-self.surprise_dist["min"])/(self.surprise_dist["max"]-self.surprise_dist["min"]))
		avg_surp = min(1,max(0,avg_surp-self.surprise_dist["min"])/(self.surprise_dist["max"]-self.surprise_dist["min"]))
		return max_surp #-avg_surp
