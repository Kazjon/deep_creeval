import argparse, pymongo, sys, pprint, random

import numpy as np
from conceptual_space import *
from deap import base, creator, tools
from operator import attrgetter

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
	if not from_visible:
		individual = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[0].tolist()[0]
	s = plausibility_score(individual, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound)
	return (s,)

def wrap_plausibility_and_surprise(individual, model=None, plausibility_dist=None, weight_by_length=True, errors_by_length=None, feature_list=None, from_visible=True, use_lower_bound=False, surprise_dist=None, surprise_depth=1):
	if not from_visible:
		individual = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[0].tolist()[0]
	p = plausibility_score(individual, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound)
	#s = surprise_score(model.positive_features_from_design_vector(individual), model, None, surprise_dist, surprise_depth)
	return (p,)

def init_GA(ndims, model, metadata, experiment, feature_prob=0.5,use_lower_bound=False):
	creator.create("MaxFitness", base.Fitness, weights=(1.0,1.0))
	creator.create("Individual", list, fitness=creator.MaxFitness)
	toolbox = base.Toolbox()
	toolbox.register("feature", lambda : random.random() < feature_prob)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature, n=ndims)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", tools.cxUniform, indpb=.5)
	toolbox.register("mutate", tools.mutFlipBit, indpb= 1. / len(metadata["fields_x"]))
	toolbox.register("select", tools.selTournament, tournsize=10)
	toolbox.register("evaluate", wrap_plausibility, model=model, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=False, errors_by_length=metadata["experiments"][experiment]["errors_by_length"], feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound)

	return toolbox

def mutReplaceGaussian(individual, indpb, mu, sigma):
	for i in range(len(individual)):
		if random.random() < indpb:
			individual[i] = random.gauss(mu, sigma)

def init_CGA(ndims, model, metadata, experiment, surprise_depth, mu=0, sigma=1, mutation_width_fraction=5.0, use_lower_bound=False):
	creator.create("MaxFitness", base.Fitness, weights=(1.0))
	creator.create("Individual", list, fitness=creator.MaxFitness)
	toolbox = base.Toolbox()
	toolbox.register("feature", lambda : random.gauss(mu,sigma))
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature, n=ndims)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", tools.cxUniform, indpb=.5)
	toolbox.register("mutate", tools.mutGaussian, indpb=1. / model.model.nhid, mu=0, sigma=sigma/float(mutation_width_fraction))
	toolbox.register("select", tools.selNSGA2)
	toolbox.register("selectTournament", tools.selTournamentDCD)
	toolbox.register("evaluate", wrap_plausibility_and_surprise, model=model, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=False, errors_by_length=metadata["experiments"][experiment]["errors_by_length"],from_visible=False, feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound, surprise_dist=metadata["experiments"][experiment]["surprise_distribution"], surprise_depth=surprise_depth)

	return toolbox

if __name__ == "__main__":
	print "Started GA_Synthesiser."
	parser = argparse.ArgumentParser(description='Use this to generate designs given an existing VAE model')
	parser.add_argument('--dataset',help="Name of the dataset to work with")
	parser.add_argument("--model",help="Path to the .pkl file containing the trained expectation model VAE.", default=None)
	parser.add_argument("--experiment",help="Name of the experiment containing the settings in use.", default=None)

	use_CGA = True

	args = parser.parse_args()
	args.experiment = sanitise_for_mongo(args.experiment)

	metadata = init_dataset(args.dataset)

	if not "surprise samples" in metadata:
		metadata["surprise_samples"] = 100000

	score_method = "plausibility"
	surprise_depth=2
	n_generations = 1000
	population_size = 1000
	crossover_prob = 0.9
	average_desired_features = 4
	lower_bound_plaus=False

	mutation_prob = 0.025

	model = init_model(args.dataset, metadata, args.model, surprise_depth, args.experiment)

	if use_CGA:
		ga_toolbox = init_CGA(model.model.nhid, model, metadata, args.experiment, surprise_depth, mu=0, sigma=1, mutation_width_fraction=5, use_lower_bound=lower_bound_plaus)
	else:
		ga_toolbox = init_GA(len(metadata["fields_x"]), model, metadata, args.experiment, feature_prob = float(average_desired_features) / len(metadata["fields_x"]), use_lower_bound=lower_bound_plaus)
	stats = tools.Statistics(key=lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)
	records = []

	pop = ga_toolbox.population(n=population_size)
	fitnesses = map(ga_toolbox.evaluate,pop)
	for ind, fit in zip(pop,fitnesses):
		ind.fitness.values = fit


	pop = ga_toolbox.select(pop, population_size) #Doesn't actually select anything, just sorts.

	bests = []
	for g in range(n_generations):
		parent_pool = ga_toolbox.selectTournament(pop, population_size)
		offspring = map(ga_toolbox.clone, parent_pool)

		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < crossover_prob:
				ga_toolbox.mate(child1,child2)

		for mutant in offspring:
			if random.random() < mutation_prob:
				ga_toolbox.mutate(mutant)

		for ind in offspring:
			ind.fitness.values = ga_toolbox.evaluate(ind)

		pop.extend(offspring)
		pop = ga_toolbox.select(pop, k=population_size)

		record = stats.compile(pop)
		print record
		if use_CGA:
			best = model.positive_features_from_design_vector(model.construct_from_hidden(np.atleast_2d(np.array(max(pop, key=attrgetter("fitness")))))[0].tolist()[0]).keys()
		else:
			best = model.positive_features_from_design_vector(max(pop, key=attrgetter("fitness"))).keys()
		bests.append(best)
		records.append(record)

		for k,ind in enumerate(sorted(pop, key=attrgetter("fitness"), reverse=True)[:10]):
			print str(k+1)+":",
			if use_CGA:
				for f in model.positive_features_from_design_vector(model.construct_from_hidden(np.atleast_2d(np.array(ind)))[0].tolist()[0]).keys():
					print f+",",
			else:
				for f,v in model.features_from_design_vector(ind).iteritems():
					if v:
						print f,
			print "({0:.4f},{1:.4f})".format(*ind.fitness.values)
		print
	print records