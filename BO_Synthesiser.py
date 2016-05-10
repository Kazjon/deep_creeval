import argparse, pymongo, sys, pprint, random

import numpy as np
from conceptual_space import *
from bayes_opt import BayesianOptimization

from Synthesis_Evaluators import plausibility_score, surprise_score

def sanitise_for_mongo(attr):
	return attr.replace(".",",")

def init_dataset(dataset):
	client = pymongo.MongoClient()
	db = client.creeval
	metadata = db.datasets.find_one({"name": dataset})

	if metadata is not None:
		if "length_distribution" not in metadata.keys():
			print "Generating distribution over number of features in each design in the dataset."
			lengths = {}
			# Grab the number of features from each design in the database
			for design in db[dataset].find():
				length  = sum(f in design.keys() for f in metadata["fields_x"])
				if length in lengths:
					lengths[length] += 1
				else:
					lengths[length] = 1
			# Pad our lengths just in case some lengths have no designs
			for i in range(max(lengths.keys())):
				if i not in lengths:
					lengths[i] = 0
			lengths = sorted(lengths.items())
			lengths = list(zip(*lengths)[1])
			total = float(sum(lengths))
			lengths = [l/total for l in lengths]
			metadata["experiments"] = lengths
			print "Saving updates to the",dataset,"dataset entry."
			db.datasets.save(metadata)
		return metadata
	else:
		print "No datasets entry for",dataset,"was found."
		sys.exit()

def init_model(dataset, metadata, model_path, surprise_depth, experiment):
	#Initialise the VAE from the given file.
	dataset_changed = False
	print "Initalising a VAE from the model file at",model_path+"."
	#model = globals()[metadata['model_class']](dataset, "data/",selected_hypers=metadata["experiments"][experiment]["best_hypers"])
	model = globals()[metadata['model_class']](dataset, "data/",selected_hypers=metadata["best_hypers"])
	if "monary_type" in metadata.keys():
		model.set_monary_type(metadata["monary_type"])
	model.load(model_path)
	model.init_model_functions()
	model.metadata = metadata
	conditional_dists_file = model_path[:-4]+"_surpdist_"+str(surprise_depth)+".csv"
	metadata["experiments"][experiment]["surprise_distribution"] = model.precalculate_conditional_dists(from_file=True,file_path=conditional_dists_file, depth=surprise_depth)
	if any(k not in metadata["experiments"][experiment].keys() for k in ["plausibility_distribution","errors_by_length","hidden_rep_averages"]):
		print "Generating distribution over plausibility for each design in the dataset."
		plausibilities, errors_by_length,hidden_rep_averages = model.get_dataset_errors(metadata, return_averages_by_length=True, return_hidden_rep_averages=True)
		print "plausibilities.shape",plausibilities.shape
		plaus_dist = {}
		plaus_dist["min"] = float(np.amin(plausibilities))
		plaus_dist["max"] = float(np.amax(plausibilities))
		plaus_dist["5%"] = float(np.percentile(plausibilities, 5))
		plaus_dist["95%"] = float(np.percentile(plausibilities, 95))
		plaus_dist["mean"] = float(np.average(plausibilities))
		print "plaus_dist",plaus_dist
		metadata["experiments"][experiment]["plausibility_distribution"] = plaus_dist
		metadata["experiments"][experiment]["errors_by_length"] = errors_by_length
		metadata["experiments"][experiment]["hidden_rep_averages"] = hidden_rep_averages
		dataset_changed = True
	if dataset_changed:
		print "Saving updates to the",dataset,"dataset entry."
		client = pymongo.MongoClient()
		db = client.creeval
		db.datasets.save(metadata)
	return model

def wrap_plausibility(individual, model=None, plausibility_dist=None, weight_by_length=True, errors_by_length=None, feature_list=None, from_visible=True, use_lower_bound=False):
	individual = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[0].tolist()[0]
	p = plausibility_score(individual, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound)
	return p

def wrap_plausibility_and_surprise(individual, model=None, plausibility_dist=None, weight_by_length=True, errors_by_length=None, feature_list=None, from_visible=True, use_lower_bound=False, surprise_dist=None, surprise_depth=1, samples=100):
	p = np.zeros(samples)
	for i in range(samples):
		individual = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[0].tolist()[0]
		p[i] = plausibility_score(individual, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound)
	print p
	print np.mean(p)
	sys.exit()
	s = surprise_score(model.positive_features_from_design_vector(individual), model, None, surprise_dist, surprise_depth)
	return np.mean(p)+s



if __name__ == "__main__":
	print "Started BO_Synthesiser."
	parser = argparse.ArgumentParser(description='Use this to generate designs given an existing VAE model')
	parser.add_argument('--dataset',help="Name of the dataset to work with")
	parser.add_argument("--model",help="Path to the .pkl file containing the trained expectation model VAE.", default=None)
	parser.add_argument("--experiment",help="Name of the experiment containing the settings in use.", default=None)

	args = parser.parse_args()
	args.experiment = sanitise_for_mongo(args.experiment)

	metadata = init_dataset(args.dataset)

	if not "surprise samples" in metadata:
		metadata["surprise_samples"] = 100000

	score_method = "plausibility"
	surprise_depth=2
	steps = 1000
	init_steps = 100
	recipes_per_step = 5
	lower_bound_plaus=False
	num_sigma_range = 5
	k=3

	gp_params = {'corr':'absolute_exponential','nugget': 1}
	model = init_model(args.dataset, metadata, args.model, surprise_depth, args.experiment)


	params = {}
	for p in range(model.model.nhid):
		params["x_"+str(p)] = (-num_sigma_range,num_sigma_range)
	bo = BayesianOptimization(lambda **param_args : wrap_plausibility_and_surprise([param_args[p] for p in param_args], model=model, plausibility_dist=metadata["experiments"][args.experiment]["plausibility_distribution"], weight_by_length=False, errors_by_length=metadata["experiments"][args.experiment]["errors_by_length"],from_visible=False, feature_list=metadata["fields_x"], use_lower_bound=lower_bound_plaus, surprise_dist=metadata["experiments"][args.experiment]["surprise_distribution"], surprise_depth=surprise_depth),params)

	bo.maximize(init_points=init_steps, n_iter=0, kappa=k)
	for step in range(steps):
		bo.maximize(init_points=0, n_iter=recipes_per_step, kappa=k, **gp_params)
		for r,v in zip(bo.X[-recipes_per_step:],bo.Y[-recipes_per_step:]):
			print model.positive_features_from_design_vector(model.construct_from_hidden(np.atleast_2d(np.array(r)))[0].tolist()[0]).keys(),"({0:.4f})".format(v)
