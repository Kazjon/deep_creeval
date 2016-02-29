__author__ = 'kazjon'
import numpy as np
import sys
from deap import creator
from math import isnan

def plausibility_score(state, model=None, plausibility_dist=None, weight_by_length=True, errors_by_length=None, feature_list=None, from_visible=True, cap_fitness =True, use_lower_bound=False):
	if type(state[0]) in [int, bool, float]:
		dict_state = {feature_list[k]:1 for k,f in enumerate(state) if f}
	else:
		dict_state = {f:1 for f in state}
	if len(dict_state) == 0:
		return 0
	ll = -model.recon_cost(dict_state, use_lower_bound=use_lower_bound)[0]
	length = len(dict_state.keys())
	if weight_by_length:
		if not isnan(errors_by_length[length]):
			ll *= errors_by_length[length] / plausibility_dist["mean"]
	if cap_fitness:
		score = 1 - min(1,(max(0,ll + plausibility_dist["max"]))/-plausibility_dist["min"])
	else:
		score = max(0,(-ll - plausibility_dist["min"])/-plausibility_dist["min"]) #1 - min(1,(max(0,ll + plausibility_dist["max"]))/-plausibility_dist["min"])
	return score


def surprise_score(state, model, end_token, surprise_dist, surprise_depth):
	state = set(state)
	if end_token is not None:
		state.remove(end_token)
	surprises = model.surprising_sets(list(state),depth_limit=surprise_depth,n_iter=10)
	surp_list = model.sorted_surprise(surprises)
	max_surp = model.max_surprise(surp_list)
	avg_surp = model.avg_surprise(surp_list)
	max_surp = min(1,max(0,max_surp-surprise_dist["min"])/(surprise_dist["max"]-surprise_dist["min"]))
	avg_surp = min(1,max(0,avg_surp-surprise_dist["min"])/(surprise_dist["max"]-surprise_dist["min"]))
	return max_surp #-avg_surp