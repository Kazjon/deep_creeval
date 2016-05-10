import argparse, pymongo, sys, random, scipy.spatial

import numpy as np
from conceptual_space import *
from deap import base, creator, tools
from operator import attrgetter

from Synthesis_Evaluators import plausibility_score, surprise_score

#Just in case we're importing this from Recipe Retrieval itself
if __name__ == "__main__":
	import Recipe_Retrieval

def sanitise_for_mongo(attr):
	return attr.replace(".",",")

def unsanitise_for_mongo(attr):
	return attr.replace(",",".")

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
	conditional_dists_file = os.path.join(os.path.split(model_path)[0],unsanitise_for_mongo(experiment)+"_surpdist_"+str(surprise_depth)+".csv")
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

	if not "surprise samples" in metadata:
		metadata["surprise_samples"] = 10000
	model.gen_surprise_samples()
	return model

def wrap_plausibility_and_surprise_holistic(population, model=None, plausibility_dist=None, weight_by_length=True, errors_by_length=None, feature_list=None, from_visible=True, use_lower_bound=False, surprise_dist=None, surprise_depth=1, min_features=3, samples=10, use_means_not_sampling=True, means_threshold=0.1, combine=False, prefs={}, max_surp_only = False, desired_surprise=-1):
	if not from_visible:
		return NotImplementedError #Haven't looked into this yet.
		if use_means_not_sampling:
			ind_means = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[1].tolist()[0]
			ind_constructed = [int(f > means_threshold * model.sample_means[i]) for i,f in enumerate(ind_means)]
			if sum(ind_constructed) < min_features:
				p=0
				s=0
			else:
				p = plausibility_score(ind_constructed, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound, use_means_not_sampling=use_means_not_sampling, prefs=prefs)
				s = surprise_score(model.positive_features_from_design_vector(ind_constructed), model, None, surprise_dist, surprise_depth)#, required_ingredients=required_ingredients)
		else:
			p = np.zeros(samples)
			s = np.zeros(samples)
			for i in range(samples):
				ind_constructed = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[0].tolist()[0]
				if sum(ind_constructed) < min_features:
					p[i]=0
					s[i]=0
				else:
					p[i] = plausibility_score(ind_constructed, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound, prefs=prefs)
					s[i] = surprise_score(model.positive_features_from_design_vector(ind_constructed), model, None, surprise_dist, surprise_depth)#, required_ingredients=required_ingredients)
			p = np.mean(p)
			s = np.mean(s)
	else:
		lls = -model.recon_cost(np.array(population), use_lower_bound=use_lower_bound)
		all_p = np.zeros(len(population))
		all_s = np.zeros(len(population))
		for k,individual in enumerate(population):
			all_p[k] = plausibility_score(individual,
			                              model=model,
			                              plausibility_dist=plausibility_dist,
			                              weight_by_length=weight_by_length,
			                              errors_by_length=errors_by_length,
			                              feature_list=feature_list,
			                              from_visible=from_visible,
			                              cap_fitness=False,
			                              use_lower_bound=use_lower_bound,
			                              ll=lls[k],
			                              prefs=prefs)
			all_s[k] = surprise_score(model.positive_features_from_design_vector(individual),
			                          model,
			                          None,
			                          surprise_dist,
			                          surprise_depth,
			                          max_surp_only=max_surp_only,
			                          desired_surprise=desired_surprise)
			if sum(individual) < min_features:
				all_p[k]*=float(sum(individual))/min_features
				#all_s[k]*=float(sum(individual))/min_features
		del lls
	if combine:
		return [(p+s,) for p,s in zip(all_p, all_s)]
	return [(p,s) for p,s in zip(all_p, all_s)]


def wrap_plausibility_and_surprise(individual, model=None, plausibility_dist=None, weight_by_length=True, errors_by_length=None, feature_list=None, from_visible=True, use_lower_bound=False, surprise_dist=None, surprise_depth=1, min_features=3, samples=10, prefs={}, use_means_not_sampling=True, means_threshold=0.1, combine=False, ll=None, recon_means=None, max_surp_only = False, desired_surprise=-1):
	if not from_visible:
		if use_means_not_sampling:
			ind_means = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[1].tolist()[0]
			ind_constructed = [int(f > means_threshold * model.sample_means[i]) for i,f in enumerate(ind_means)]
			if sum(ind_constructed) < min_features:
				p=0
				s=0
			else:
				p = plausibility_score(ind_constructed, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound, use_means_not_sampling=use_means_not_sampling, prefs=prefs)
				s = surprise_score(model.positive_features_from_design_vector(ind_constructed), model, None, surprise_dist, surprise_depth, max_surp_only=max_surp_only, desired_surprise=desired_surprise)
		else:
			p = np.zeros(samples)
			s = np.zeros(samples)
			for i in range(samples):
				ind_constructed = model.construct_from_hidden(np.atleast_2d(np.array(individual)))[0].tolist()[0]
				if sum(ind_constructed) < min_features:
					p[i]=0
					s[i]=0
				else:
					p[i] = plausibility_score(ind_constructed, model=model, plausibility_dist=plausibility_dist, weight_by_length=weight_by_length, errors_by_length=errors_by_length, feature_list=feature_list, from_visible=from_visible, cap_fitness=False, use_lower_bound=use_lower_bound, prefs=prefs)
					s[i] = surprise_score(model.positive_features_from_design_vector(ind_constructed), model, None, surprise_dist, surprise_depth, max_surp_only=max_surp_only, desired_surprise=desired_surprise)#, required_ingredients=required_ingredients)
			p = np.mean(p)
			s = np.mean(s)
	else:
		p = plausibility_score(individual,
		                       model=model,
		                       plausibility_dist=plausibility_dist,
		                       weight_by_length=weight_by_length,
		                       errors_by_length=errors_by_length,
		                       feature_list=feature_list,
		                       from_visible=from_visible,
		                       cap_fitness=False,
		                       use_lower_bound=use_lower_bound,
		                       ll=ll, recon_means=recon_means,
		                       prefs=prefs)
		s = surprise_score(model.positive_features_from_design_vector(individual),
		                   model,
		                   None,
		                   surprise_dist,
		                   surprise_depth,
		                   prefs=prefs,
		                   max_surp_only=max_surp_only,
		                   desired_surprise=desired_surprise)
		if sum(individual) < min_features:
			p*=float(sum(individual))/min_features
			#s*=float(sum(individual))/min_features
	if combine:
		return (p+s,)
	return (p,s)

def init_GA(ndims, model, metadata, experiment, feature_prob=0.5, min_features=3, use_lower_bound=False, required_ingredients=[], single_fitness=False, forbidden_ingredients=[]):
	if single_fitness:
		creator.create("MaxFitness", base.Fitness, weights=(1.0,))
	else:
		creator.create("MaxFitness", base.Fitness, weights=(1.0,1.0))
	creator.create("Individual", list, fitness=creator.MaxFitness)
	toolbox = base.Toolbox()
	toolbox.register("feature", lambda : random.random() < feature_prob)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature, n=ndims)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", tools.cxUniform, indpb=.5)
	toolbox.register("mutate", tools.mutFlipBit, indpb= 1. / len(metadata["fields_x"]))

	if single_fitness:
		toolbox.register("select", tools.selTournament, tournsize=2)
	else:
		toolbox.register("select", tools.selNSGA2)
		toolbox.register("selectTournament", tools.selTournamentDCD)
	#toolbox.register("evaluate", wrap_plausibility, model=model, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=False, errors_by_length=metadata["experiments"][experiment]["errors_by_length"], feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound)
	toolbox.register("evaluate", wrap_plausibility_and_surprise, model=model, min_features=min_features, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=True, errors_by_length=metadata["experiments"][experiment]["errors_by_length"],from_visible=True, feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound, surprise_dist=metadata["experiments"][experiment]["surprise_distribution"], surprise_depth=surprise_depth, required_ingredients= required_ingredients, combine=single_fitness, forbidden_ingredients=forbidden_ingredients)
	toolbox.register("evaluateHolistic", wrap_plausibility_and_surprise_holistic, model=model, min_features=min_features, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=True, errors_by_length=metadata["experiments"][experiment]["errors_by_length"],from_visible=True, feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound, surprise_dist=metadata["experiments"][experiment]["surprise_distribution"], surprise_depth=surprise_depth, required_ingredients= required_ingredients, combine=single_fitness, forbidden_ingredients=forbidden_ingredients)

	return toolbox

def init_CGA(ndims, model, metadata, experiment, surprise_depth, mu=0, sigma=1, mutation_width_fraction=5.0, use_lower_bound=False, required_ingredients=[], use_means_not_sampling=False, means_threshold=0.1, single_fitness=False, forbidden_ingredients=[]):
	creator.create("MaxFitness", base.Fitness, weights=(1.0,1.0))
	creator.create("Individual", list, fitness=creator.MaxFitness)
	toolbox = base.Toolbox()
	toolbox.register("feature", lambda : random.gauss(mu,sigma))
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature, n=ndims)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", tools.cxUniform, indpb=.5)
	toolbox.register("mutate", tools.mutGaussian, indpb=1. / model.model.nhid, mu=0, sigma=sigma/float(mutation_width_fraction))
	toolbox.register("select", tools.selNSGA2)
	toolbox.register("selectTournament", tools.selTournamentDCD)
	toolbox.register("evaluate", wrap_plausibility_and_surprise, model=model, plausibility_dist=metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=True, errors_by_length=metadata["experiments"][experiment]["errors_by_length"],from_visible=False, feature_list=metadata["fields_x"], use_lower_bound=use_lower_bound, surprise_dist=metadata["experiments"][experiment]["surprise_distribution"], surprise_depth=surprise_depth, required_ingredients= required_ingredients, use_means_not_sampling=use_means_not_sampling, means_threshold=means_threshold)

	return toolbox

class GA_Synthesiser():

	def __init__(self, dataset, model, experiment, use_CGA=False, evaluate_holistic=True, surprise_depth=True, population_size=1000, crossover_prob=1, mutation_prob=0.1, average_desired_features=8, print_thresh = 0.1, lower_bound_plaus=True, single_fitness=True, use_means_not_sampling=False, means_threshold=0.1, required_ingredients=[], forbidden_ingredients=[], preferred_ingredients=[], disliked_ingredients=[], max_surp_only=False, desired_surprise=1.0):
		self.dataset = dataset
		self.experiment = experiment
		self.use_CGA = use_CGA
		self.print_thresh = print_thresh
		self.use_means_not_sampling = use_means_not_sampling
		self.means_threshold = means_threshold
		self.evaluate_holistic = evaluate_holistic
		self.population_size = population_size
		self.crossover_prob = crossover_prob
		self.mutation_prob = mutation_prob
		self.prefs = {"required": required_ingredients, "forbidden": forbidden_ingredients, "preferred": preferred_ingredients, "disliked": disliked_ingredients}
		self.single_fitness = single_fitness
		self.lower_bound_plaus = lower_bound_plaus
		self.max_surp_only = max_surp_only
		self.desired_surprise = desired_surprise

		self.metadata = self.init_dataset()
		self.model = self.init_model(model, surprise_depth, experiment)

		if use_CGA:
			self.toolbox = self.init_CGA(args.experiment, surprise_depth, mu=0, sigma=1, mutation_width_fraction=2, use_lower_bound=lower_bound_plaus, use_means_not_sampling=use_means_not_sampling, means_threshold=means_threshold, single_fitness=single_fitness)
		else:
			self.toolbox = self.init_GA(args.experiment, min_features=average_desired_features, feature_prob = float(average_desired_features) / len(self.metadata["fields_x"]), use_lower_bound=lower_bound_plaus, single_fitness=single_fitness)
		self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
		self.stats.register("avg", np.mean)
		self.stats.register("std", np.std)
		self.stats.register("min", np.min)
		self.stats.register("max", np.max)
		self.records = []
		self.bests = []

		print "-- Initialising GA."
		self.pop = self.toolbox.population(n=population_size)
		print "  -- Beginning fitness eval."
		if evaluate_holistic:
			fitnesses = self.toolbox.evaluateHolistic(self.pop)
		else:
			fitnesses = map(self.toolbox.evaluate,self.pop)

		for ind, fit in zip(self.pop,fitnesses):
			ind.fitness.values = fit
		print "  -- Fitness eval complete."

		if not single_fitness:
			self.pop = self.toolbox.select(self.pop, population_size) #Doesn't actually select anything, just sorts.

		print "-- GA Initialisation complete."

	def init_dataset(self):
		client = pymongo.MongoClient()
		db = client.creeval
		metadata = db.datasets.find_one({"name": self.dataset})

		if metadata is not None:
			if "length_distribution" not in metadata.keys():
				print "Generating distribution over number of features in each design in the dataset."
				lengths = {}
				# Grab the number of features from each design in the database
				for design in db[self.dataset].find():
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
				print "Saving updates to the",self.dataset,"dataset entry."
				db.datasets.save(metadata)
			return metadata
		else:
			print "No datasets entry for",self.dataset,"was found."
			sys.exit()

	def init_model(self, model_path, surprise_depth, experiment):
		#Initialise the VAE from the given file.
		dataset_changed = False
		print "Initalising a VAE from the model file at",model_path+"."
		#model = globals()[metadata['model_class']](dataset, "data/",selected_hypers=metadata["experiments"][experiment]["best_hypers"])
		model = globals()[self.metadata['model_class']](self.dataset, "data/",selected_hypers=self.metadata["best_hypers"])
		if "monary_type" in self.metadata.keys():
			model.set_monary_type(self.metadata["monary_type"])
		model.load(model_path)
		model.init_model_functions()
		model.metadata = self.metadata
		conditional_dists_file = os.path.join(os.path.split(model_path)[0],unsanitise_for_mongo(experiment)+"_surpdist_"+str(surprise_depth)+".csv")
		self.metadata["experiments"][experiment]["surprise_distribution"] = model.precalculate_conditional_dists(from_file=True,file_path=conditional_dists_file, depth=surprise_depth)
		if any(k not in self.metadata["experiments"][experiment].keys() for k in ["plausibility_distribution","errors_by_length","hidden_rep_averages"]):
			print "Generating distribution over plausibility for each design in the dataset."
			plausibilities, errors_by_length,hidden_rep_averages = model.get_dataset_errors(self.metadata, return_averages_by_length=True, return_hidden_rep_averages=True)
			print "plausibilities.shape",plausibilities.shape
			plaus_dist = {}
			plaus_dist["min"] = float(np.amin(plausibilities))
			plaus_dist["max"] = float(np.amax(plausibilities))
			plaus_dist["5%"] = float(np.percentile(plausibilities, 5))
			plaus_dist["95%"] = float(np.percentile(plausibilities, 95))
			plaus_dist["mean"] = float(np.average(plausibilities))
			print "plaus_dist",plaus_dist
			print "plaus stdev:",np.std(plausibilities)
			self.metadata["experiments"][experiment]["plausibility_distribution"] = plaus_dist
			self.metadata["experiments"][experiment]["errors_by_length"] = errors_by_length
			self.metadata["experiments"][experiment]["hidden_rep_averages"] = hidden_rep_averages
			dataset_changed = True
		if dataset_changed:
			print "Saving updates to the",self.dataset,"dataset entry."
			client = pymongo.MongoClient()
			db = client.creeval
			db.datasets.save(self.metadata)

		if not "surprise samples" in self.metadata:
			self.metadata["surprise_samples"] = 10000
		model.gen_surprise_samples()
		return model

	def init_GA(self, experiment, feature_prob=0.5, min_features=3, use_lower_bound=False, single_fitness=False):

		ndims = len(self.metadata["fields_x"])
		if single_fitness:
			creator.create("MaxFitness", base.Fitness, weights=(1.0,))
		else:
			creator.create("MaxFitness", base.Fitness, weights=(1.0,1.0))
		creator.create("Individual", list, fitness=creator.MaxFitness)
		toolbox = base.Toolbox()
		toolbox.register("feature", lambda : random.random() < feature_prob)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature, n=ndims)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mate", tools.cxUniform, indpb=.5)
		toolbox.register("mutate", tools.mutFlipBit, indpb= 1. / len(self.metadata["fields_x"]))

		if single_fitness:
			toolbox.register("select", tools.selTournament, tournsize=2)
		else:
			toolbox.register("select", tools.selNSGA2)
			toolbox.register("selectTournament", tools.selTournamentDCD)
		toolbox.register("evaluate", wrap_plausibility_and_surprise,
		                 model=self.model,
		                 min_features=min_features,
		                 plausibility_dist=self.metadata["experiments"][experiment]["plausibility_distribution"],
		                 weight_by_length=True,
		                 errors_by_length=self.metadata["experiments"][experiment]["errors_by_length"],
		                 from_visible=True,
		                 feature_list=self.metadata["fields_x"],
		                 use_lower_bound=use_lower_bound,
		                 surprise_dist=self.metadata["experiments"][experiment]["surprise_distribution"],
		                 surprise_depth=surprise_depth,
		                 prefs= self.prefs,
		                 combine=single_fitness,
		                 max_surp_only = self.max_surp_only,
		                 desired_surprise=self.desired_surprise)

		toolbox.register("evaluateSeparately", wrap_plausibility_and_surprise,
		                 model=self.model,
		                 min_features=min_features,
		                 plausibility_dist=self.metadata["experiments"][experiment]["plausibility_distribution"],
		                 weight_by_length=True,
		                 errors_by_length=self.metadata["experiments"][experiment]["errors_by_length"],
		                 from_visible=True,
		                 feature_list=self.metadata["fields_x"],
		                 use_lower_bound=use_lower_bound,
		                 surprise_dist=self.metadata["experiments"][experiment]["surprise_distribution"],
		                 surprise_depth=surprise_depth,
		                 prefs= self.prefs,
		                 combine=False,
		                 max_surp_only = self.max_surp_only)

		toolbox.register("evaluateHolistic", wrap_plausibility_and_surprise_holistic,
		                 model=self.model,
		                 min_features=min_features,
		                 plausibility_dist=self.metadata["experiments"][experiment]["plausibility_distribution"],
		                 weight_by_length=True,
		                 errors_by_length=self.metadata["experiments"][experiment]["errors_by_length"],
		                 from_visible=True,
		                 feature_list=self.metadata["fields_x"],
		                 use_lower_bound=use_lower_bound,
		                 surprise_dist=self.metadata["experiments"][experiment]["surprise_distribution"],
		                 surprise_depth=surprise_depth,
		                 prefs= self.prefs,
		                 combine=single_fitness,
		                 max_surp_only = self.max_surp_only,
		                 desired_surprise=self.desired_surprise)

		return toolbox

	def init_CGA(self, experiment, surprise_depth, mu=0, sigma=1, mutation_width_fraction=5.0, use_lower_bound=False, use_means_not_sampling=False, means_threshold=0.1, single_fitness=False):

		ndims = self.model.model.nhid
		creator.create("MaxFitness", base.Fitness, weights=(1.0,1.0))
		creator.create("Individual", list, fitness=creator.MaxFitness)
		toolbox = base.Toolbox()
		toolbox.register("feature", lambda : random.gauss(mu,sigma))
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature, n=ndims)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mate", tools.cxUniform, indpb=.5)
		toolbox.register("mutate", tools.mutGaussian, indpb=1. / self.model.model.nhid, mu=0, sigma=sigma/float(mutation_width_fraction))
		toolbox.register("select", tools.selNSGA2)
		toolbox.register("selectTournament", tools.selTournamentDCD)
		toolbox.register("evaluate", wrap_plausibility_and_surprise, model=self.model, plausibility_dist=self.metadata["experiments"][experiment]["plausibility_distribution"], weight_by_length=True, errors_by_length=self.metadata["experiments"][experiment]["errors_by_length"],from_visible=False, feature_list=self.metadata["fields_x"], use_lower_bound=use_lower_bound, surprise_dist=self.metadata["experiments"][experiment]["surprise_distribution"], surprise_depth=surprise_depth, prefs= self.prefs, use_means_not_sampling=use_means_not_sampling, means_threshold=means_threshold, desired_surprise=self.desired_surprise)

		return toolbox

	def run_generation(self, print_best=0, print_non_dom_only=True):
		if self.single_fitness:
			parent_pool = self.toolbox.select(self.pop, len(self.pop))
		else:
			parent_pool = self.toolbox.selectTournament(self.pop, population_size)
		offspring = map(self.toolbox.clone, parent_pool)

		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < crossover_prob:
				self.toolbox.mate(child1,child2)

		for mutant in offspring:
			if random.random() < mutation_prob:
				self.toolbox.mutate(mutant)

		print "  -- Beginning fitness eval."
		if self.evaluate_holistic:
			fitnesses = self.toolbox.evaluateHolistic(offspring)
		else:
			fitnesses = map(self.toolbox.evaluate,offspring)

		for ind, fit in zip(offspring,fitnesses):
			ind.fitness.values = fit
		print "  -- Fitness eval complete."

		#for ind in offspring:
		#	ind.fitness.values = ga_toolbox.evaluate(ind)

		if single_fitness:
			self.pop = offspring
		else:
			self.pop.extend(offspring)
			self.pop = self.toolbox.select(self.pop, k=self.population_size)
		if print_best > 0:
			self.print_best(self.pop,print_best, non_dom=print_non_dom_only, recalc=True)

		record = self.stats.compile(self.pop)
		print "-- Fitness stats:",record
		if use_CGA:
			best = self.model.positive_features_from_design_vector(self.model.construct_from_hidden(np.atleast_2d(np.array(max(self.pop, key=attrgetter("fitness")))))[0].tolist()[0]).keys() + self.prefs["required"]
		else:
			best = self.model.positive_features_from_design_vector(max(self.pop, key=attrgetter("fitness"))).keys() + self.prefs["required"]
		self.bests.append(best)
		self.records.append(record)



	def print_best(self, pop, num_to_print, recalc=False, non_dom=False, prevent_dupes=True, print_dominates=False, print_crowding=False):
		if non_dom:
			to_print = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
		else:
			#to_print = pop[:num_to_print]
			to_print = [item for sublist in tools.sortNondominated(pop, len(pop)) for item in sublist]
			to_print = to_print[:num_to_print]


		dupe = False
		for k,ind in enumerate(sorted(to_print, key=attrgetter("fitness"), reverse=True)): #enumerate(sorted(self.pop, key=attrgetter("fitness"), reverse=True)[:num_to_print])
			if k > 0 and ind == to_print[k-1]:
				dupe = True
			else:
				if dupe:
					print
				dupe = False
			if not (prevent_dupes and dupe):
				print str(k+1)+":",
				if self.use_CGA:
					#This older version grabs a single sample from the decoder network
					#for f in model.positive_features_from_design_vector(model.construct_from_hidden(np.atleast_2d(np.array(ind)))[0].tolist()[0]).keys():

					#This new version grabs and prints the means instead.
					recipe = self.model.features_from_design_vector(self.model.construct_from_hidden(np.atleast_2d(np.array(ind)))[1].tolist()[0]).items()
					#for i in required_ingredients:
					#	print i+":REQ,",
					for f,v in sorted(recipe,key=lambda x:x[1],reverse=True):
						if v>self.print_thresh:
							if self.use_means_not_sampling:
								print f[2:]+",",
							else:
								print f[2:]+":{0:.2f},".format(v),
				else:
					for f,v in self.model.features_from_design_vector(ind).iteritems():
						if v:
							print f[2:]+",",
				print "(",
				if recalc:
					for v in self.toolbox.evaluateSeparately(ind):
						print "{0:.3f}".format(v),
				else:
					for v in ind.fitness.values:
						print "{0:.3f}".format(v),
				print ")",
				if print_crowding:
					print "Crowding: {0:.3f},".format(ind.fitness.crowding_dist),
				if print_dominates and k+1 < len(to_print):
					print "Dominates next:",ind.fitness.dominates(to_print[k+1].fitness)
				else:
					print
			else:
				print ".",
		print

	def print_records(self):
		print self.records


	def retrieve_and_adapt_best(self, num_to_adapt, retrieval_cats, retrieval_tree, recalc=False, num_to_return=3, surprise_proximity_thresh = 0.25):
		print
		print "  -- Retrieving and adapting."
		client = pymongo.MongoClient()
		db = client.creeval
		coll = db[self.metadata["name"]]

		for k,ind in enumerate(sorted(self.pop, key=attrgetter("fitness"), reverse=True)[:num_to_adapt]):
			print str(k+1)+":",
			if self.use_CGA:

				raise NotImplementedError()
				#This older version grabs a single sample from the decoder network
				#for f in model.positive_features_from_design_vector(model.construct_from_hidden(np.atleast_2d(np.array(ind)))[0].tolist()[0]).keys():

				#This new version grabs and prints the means instead.
				recipe = self.model.features_from_design_vector(self.model.construct_from_hidden(np.atleast_2d(np.array(ind)))[1].tolist()[0]).items()
				#for i in required_ingredients:
				#	print i+":REQ,",
				for f,v in sorted(recipe,key=lambda x:x[1],reverse=True):
					if v>self.print_thresh:
						if self.use_means_not_sampling:
							print f[2:]+",",
						else:
							print f[2:]+":{0:.2f},".format(v),
			else:
				recipe=self.model.positive_features_from_design_vector(ind)
				for f in recipe:
					print f[2:]+",",
			print "(",
			if recalc:
				for v in self.toolbox.evaluateSeparately(ind):
					print "{0:.3f}".format(v),
			else:
				for v in ind.fitness.values:
					print "{0:.3f}".format(v),
			print ")"

			targets, target_ids = Recipe_Retrieval.find_closest_recipe(recipe, self.model, retrieval_tree, num_to_return=num_to_return)
			for target,target_id in zip(targets,target_ids):
				cursor = coll.find_one({"_id": target_id})
				Recipe_Retrieval.adapt_recipe(recipe, target, cursor["raw_text"], self.model, self.metadata, self.experiment, retrieval_cats, desired_surprise=self.desired_surprise, max_surp_only=self.max_surp_only, surprise_proximity_thresh=surprise_proximity_thresh)
		print

if __name__ == "__main__":
	print "Started GA_Synthesiser."
	parser = argparse.ArgumentParser(description='Use this to generate designs given an existing VAE model')
	parser.add_argument('--dataset',help="Name of the dataset to work with")
	parser.add_argument("--model",help="Path to the .pkl file containing the trained expectation model VAE.", default=None)
	parser.add_argument("--experiment",help="Name of the experiment containing the settings in use.", default=None)



	args = parser.parse_args()
	args.experiment = sanitise_for_mongo(args.experiment)

	use_CGA = False
	evaluate_holistic = True
	surprise_depth=2
	max_surp_only = True
	n_generations = 26
	retrieval_gens = 25
	population_size = 10000
	crossover_prob = 0.9
	average_desired_features = 6
	lower_bound_plaus=False
	single_fitness=False
	desired_surprise = 0.8

	#For determining whether we sample from the distribution over ingredients, or just take high mean values.
	use_means_not_sampling=True
	means_threshold = 0.1

	mutation_prob = 0.1
	print_thresh = 0.1


	cases = []
	cases.append({"name":"Picky Kid","required":["i_cheese"],"forbidden":["i_hot sauce", "i_ground chillies", "i_chillies", "i_turmeric", "i_peanut butter", "i_spinach", "i_olives","i_vinegar"],"surprise_goal":0.3})
	cases.append({"name":"Vegetarian","required":["i_beans"],"forbidden":["i_fish", "i_beef", "i_chicken", "i_prawns", "i_bacon","i_sausage"],"surprise_goal":0.6})
	cases.append({"name":"Halal Foodie","required":["i_chicken"],"forbidden":["i_pork", "i_sausage", "i_bacon", "i_red wine", "i_white wine", "i_sherry"],"surprise_goal":0.9})
	#cases.append({"name":"Average Joe","required":["i_beef","i_bread"],"forbidden":["i_mint", "i_coriander", "i_soy sauce", "i_cumin"],"surprise_goal":0.5})
	#cases.append({"name":"Sweet Tooth","required":["i_sugar","i_butter"],"forbidden":["i_peanuts", "i_peanut oil"],"surprise_goal":0.75})

	#Nadia's User 1
	#required_ingredients = ["i_spinach", "i_chicken"]
	#forbidden_ingredients = ["i_hot sauce", "i_dill", "i_pasta", "i_peanut butter", "i_peanut oil"]
	#Nadia's User 2
	#required_ingredients = ["i_rice"]
	#forbidden_ingredients = ["i_fish", "i_beef", "i_chicken", "i_prawns", "i_stock", "i_bacon", "i_pork", "i_sausage", "i_eggs", "i_flour"]
	preferred_ingredients = []#["i_zucchini", "i_mushrooms", "i_onions"]
	disliked_ingredients = []#["I_vegetable oil", "i_beans"]


	random.seed(0)
	np.random.seed(0)

	for case in cases:
		required_ingredients = case["required"]
		forbidden_ingredients = case["forbidden"]
		desired_surprise = case["surprise_goal"]

		gas = GA_Synthesiser(args.dataset, args.model, args.experiment,
		                     use_CGA=use_CGA,
		                     evaluate_holistic=evaluate_holistic,
		                     surprise_depth=surprise_depth,
		                     population_size=population_size,
		                     crossover_prob=crossover_prob,
		                     average_desired_features=average_desired_features,
		                     print_thresh=print_thresh,
		                     use_means_not_sampling=use_means_not_sampling,
		                     means_threshold=means_threshold,
		                     mutation_prob=mutation_prob,
		                     required_ingredients=required_ingredients,
		                     forbidden_ingredients=forbidden_ingredients,
		                     preferred_ingredients=preferred_ingredients,
		                     disliked_ingredients=disliked_ingredients,
		                     lower_bound_plaus=lower_bound_plaus,
		                     single_fitness=single_fitness,
		                     max_surp_only=max_surp_only,
		                     desired_surprise=desired_surprise
							)

		retrieval_cats = Recipe_Retrieval.init_substitution(gas.metadata)
		retrieval_tree = Recipe_Retrieval.init_recipe_retrieval(gas.model, gas.metadata, os.path.split(args.model)[0], args.experiment)

		for i in range(n_generations):
			print "Case:",case["name"]+",  Generation:",str(i)
			gas.run_generation(print_best=100, print_non_dom_only=False)
			if i >= retrieval_gens:
				gas.retrieve_and_adapt_best(10, retrieval_cats, retrieval_tree, recalc=single_fitness, surprise_proximity_thresh = 0.3)

		del gas
		del retrieval_cats
		del retrieval_tree