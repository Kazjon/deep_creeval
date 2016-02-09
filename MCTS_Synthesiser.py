import argparse, pymongo, sys, pprint

import numpy as np
from conceptual_space import *
from MCTS_DesignSpace import MCTSDesignSpace
from MCTS import MonteCarlo

__author__ = 'kazjon'

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
			metadata["length_distribution"] = lengths
			print "Saving updates to the",dataset,"dataset entry."
			db.datasets.save(metadata)
		return metadata
	else:
		print "No datasets entry for",dataset,"was found."
		sys.exit()

def init_model(dataset, metadata, model_path, surprise_depth):
	#Initialise the VAE from the given file.
	dataset_changed = False
	print "Initalising a VAE from the model file at",model_path+"."
	model = globals()[metadata['model_class']](dataset, "data/",selected_hypers=metadata["best_hypers"])
	if "monary_type" in metadata.keys():
		model.set_monary_type(metadata["monary_type"])
	model.load(model_path)
	model.init_model_functions()
	model.metadata = metadata
	conditional_dists_file = model_path[:-4]+"_surpdist_"+str(surprise_depth)+".csv"
	metadata["surprise_distribution"] = model.precalculate_conditional_dists(from_file=True,file_path=conditional_dists_file, depth=surprise_depth)
	if "plausibility_distribution" not in metadata:
		print "Generating distribution over plausibility for each design in the dataset."
		plausibilities = model.get_dataset_errors(metadata)
		print "plausibilities.shape",plausibilities.shape
		plaus_dist = {}
		plaus_dist["min"] = float(np.amin(plausibilities))
		plaus_dist["max"] = float(np.amax(plausibilities))
		plaus_dist["5%"] = float(np.percentile(plausibilities, 5))
		plaus_dist["95%"] = float(np.percentile(plausibilities, 95))
		plaus_dist["mean"] = float(np.average(plausibilities))
		print "plaus_dist",plaus_dist
		metadata["plausibility_distribution"] = plaus_dist
		dataset_changed = True
	if dataset_changed:
		print "Saving updates to the",dataset,"dataset entry."
		client = pymongo.MongoClient()
		db = client.creeval
		db.datasets.save(metadata)
	return model


if __name__ == "__main__":
	print "Started MCTS_Synthesiser."
	parser = argparse.ArgumentParser(description='Use this to generate designs given an existing VAE model')
	parser.add_argument('--dataset',help="Name of the dataset to work with")
	parser.add_argument("--model",help="Path to the .pkl file containing the trained expectation model VAE.", default=None)

	args = parser.parse_args()

	metadata = init_dataset(args.dataset)

	if not "surprise samples" in metadata:
		metadata["surprise_samples"] = 10000


	min_ing = 1
	max_ing = 6
	seconds_per_action = 60
	C = 2
	keep_best = 10
	score_method = "surprise"
	surprise_depth=2


	model = init_model(args.dataset, metadata, args.model, surprise_depth)
	design_space = MCTSDesignSpace(model,metadata["fields_x"], plausibility_distribution=metadata["plausibility_distribution"], length_distribution=metadata["length_distribution"], surprise_distribution=metadata["surprise_distribution"], min_moves=min_ing, max_moves=max_ing, score_method=score_method, surprise_depth=surprise_depth)
	mcts = MonteCarlo(design_space, max_moves=max_ing, time=seconds_per_action, C=C, heavy_playouts=True, keep_best=keep_best)
	mcts.start()
	for _ in range(max_ing):
		ing = mcts.get_play()
		mcts.update(ing)
		print "Ingredient selected:",ing
		print "Current state:",mcts.states[-1]
		print "Best",keep_best,"recipes found so far:"
		for i,r in enumerate(mcts.get_best()):
			print str(i+1)+": ",r[0],"score:",r[1]
		if ing == design_space.end_token:
			break