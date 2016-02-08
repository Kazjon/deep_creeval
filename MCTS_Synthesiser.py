import argparse, pymongo, sys

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

def init_model(dataset, metadata, model_path):
	#Initialise the VAE from the given file.
	print "Initalising a VAE from the model file at",model_path+"."
	model = globals()[metadata['model_class']](dataset, "data/",selected_hypers=metadata["best_hypers"])
	if "monary_type" in metadata.keys():
		model.set_monary_type(metadata["monary_type"])
	model.load(model_path)
	model.init_model_functions()
	model.metadata = metadata
	if "plausibility_distribution" not in metadata.keys():
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
	model = init_model(args.dataset, metadata, args.model)

	min_ing = 4
	max_ing = 12
	seconds_per_action = 60

	design_space = MCTSDesignSpace(model,metadata["fields_x"], metadata["plausibility_distribution"], min_moves=min_ing, max_moves=max_ing)
	mcts = MonteCarlo(design_space, length_distribution=metadata["length_distribution"], max_moves=max_ing, time=seconds_per_action)
	mcts.start()
	for _ in range(max_ing):
		ing = mcts.get_play()
		mcts.update(ing)
		print "Ingredient selected:",ing
		print "Current state:",mcts.states[-1]
		if ing == design_space.end_token:
			break