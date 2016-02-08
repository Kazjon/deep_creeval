from __future__ import division
import datetime, random, math, sys


class MonteCarlo(object):
	def __init__(self, design_space, **kwargs):
		# Takes an instance of an MCTSDesignSpace and optionally some keyword
		# arguments.  Initializes the list of game states and the statistics tables.
		self.design_space = design_space
		self.states = []
		seconds = kwargs.get('time', 60)
		self.calculation_time = datetime.timedelta(seconds=seconds)
		self.max_moves = kwargs.get('max_moves', 30)
		self.min_moves = kwargs.get('min_moves', 3)
		self.length_distribution = kwargs.get("length_distribution")
		self.C = kwargs.get('C', 1.4)
		self.scores = {}
		self.plays = {}

	def start(self):
		self.states.append(self.design_space.start())

	def update(self, feature):
		self.states.append(self.design_space.next_state(self.states[-1],feature))

	def get_play(self):
		self.max_depth = 0
		state = self.states[-1]
		legal = self.design_space.legal_plays(self.states[:])

		# Bail out early if there is no real random.choice to be made.
		if not legal:
			print "No legal moves found!"
			sys.exit()
		if len(legal) == 1:
			return legal[0]

		games = 0
		begin = datetime.datetime.utcnow()
		while datetime.datetime.utcnow() - begin < self.calculation_time:
			self.run_simulation()
			games += 1

		moves_states = [(p, self.design_space.next_state(state, p)) for p in legal]

		# Display the number of calls of `run_simulation` and the
		# time elapsed.
		print games, datetime.datetime.utcnow() - begin

		# Pick the move with the highest percentage of wins.
		percent_wins, move = max((self.scores.get(S, 0) / self.plays.get(S, 1), p) for p, S in moves_states)

		# Display the stats for each possible play.
		for x in sorted(
			((100 * self.scores.get(S, 0) / self.plays.get(S, 1),
			self.scores.get(S, 0),
			self.plays.get(S, 0), p)
			for p, S in moves_states), reverse=True
		):
			print "{3}: {0:.2f}% ({1:.2f} / {2})".format(*x)

		print "Maximum depth searched:", self.max_depth

		return move

	def run_simulation(self):
		plays, scores = self.plays, self.scores

		visited_states = set()
		states_copy = self.states[:]
		state = states_copy[-1]

		expand = True
		for t in xrange(1, self.max_moves + 1):
			legal = self.design_space.legal_plays(states_copy)
			moves_states = [(p, self.design_space.next_state(state, p)) for p in legal]

			if all(plays.get(S) for p, S in moves_states):
				# If we have stats on all of the legal moves here, use them.
				log_total = math.log(
					sum(plays[S] for p, S in moves_states)
				)
				value, move, state = max(
					((scores[S] / plays[S]) + self.C * math.sqrt(log_total / plays[S]), p, S) for p, S in moves_states
				)
			else:
				# Otherwise, just make an arbitrary decision.
				move, state = random.choice(moves_states)

			if type(state) is tuple:
				states_copy.append(state)
			else:
				states_copy.append((state,))

			# `player` here and below refers to the player
			# who moved into that particular state.
			if expand and state not in plays:
				expand = False
				plays[state] = 0
				scores[state] = 0
				if t > self.max_depth:
					self.max_depth = t

			visited_states.add(state)

			score = self.design_space.score(states_copy)
			if score is not -1:
				break

		for state in visited_states:
			if state not in plays:
				continue
			plays[state] += 1
			if score is not -1:
				scores[state] += score