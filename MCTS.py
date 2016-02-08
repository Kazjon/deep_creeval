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
		self.wins = {}
		self.plays = {}

	def start(self):
		self.states.append(self.design_space.start())

	def update(self, feature):
		self.states.append(self.design_space.next_state(self.states[-1],feature))

	def get_play(self):
		self.max_depth = 0
		state = self.states[-1]
		player = self.design_space.current_player(state)
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
		percent_wins, move = max(
			(self.wins.get((player, S), 0) /
			 self.plays.get((player, S), 1),
			 p)
			for p, S in moves_states
		)

		# Display the stats for each possible play.
		for x in sorted(
			((100 * self.wins.get((player, S), 0) /
				self.plays.get((player, S), 1),
				self.wins.get((player, S), 0),
				self.plays.get((player, S), 0), p)
				for p, S in moves_states),
			reverse=True
		):
			print "{3}: {0:.2f}% ({1} / {2})".format(*x)

		print "Maximum depth searched:", self.max_depth

		return move

	def run_simulation(self):
		plays, wins = self.plays, self.wins

		visited_states = set()
		states_copy = self.states[:]
		state = states_copy[-1]
		player = self.design_space.current_player(state)

		expand = True
		for t in xrange(1, self.max_moves + 1):
			legal = self.design_space.legal_plays(states_copy)
			moves_states = [(p, self.design_space.next_state(state, p)) for p in legal]

			if all(plays.get((player, S)) for p, S in moves_states):
				# If we have stats on all of the legal moves here, use them.
				log_total = math.log(
					sum(plays[(player, S)] for p, S in moves_states)
				)
				value, move, state = max(
					((wins[(player, S)] / plays[(player, S)]) + self.C * math.sqrt(log_total / plays[(player, S)]), p, S) for p, S in moves_states
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
			if expand and (player, state) not in plays:
				expand = False
				plays[(player, state)] = 0
				wins[(player, state)] = 0
				if t > self.max_depth:
					self.max_depth = t

			visited_states.add((player, state))

			player = self.design_space.current_player(state)
			winner = self.design_space.winner(states_copy)
			if winner:
				break

		for player, state in visited_states:
			if (player, state) not in plays:
				continue
			plays[(player, state)] += 1
			if player == winner:
				wins[(player, state)] += 1