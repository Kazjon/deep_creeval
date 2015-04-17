import pymongo, argparse, csv, sys
from pymongo import MongoClient

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Use this to import data for use by creeval')
	parser.add_argument('-i','--input',help="Input csv file",required=True)
	parser.add_argument('-o','--output',help='Output collection in DB',required=True)
	args = parser.parse_args()
	client = MongoClient()
	db = client.creeval
	datasets = db.datasets
	data = db[args.output]

	with open(args.input) as csvfile:
		reader = csv.DictReader(csvfile)
		datasets.insert({"fields": reader.fieldnames})
		for row in reader:
			print unicode(row['name'],'utf-8',errors='ignore')
			row['name'] = unicode(row['name'],'utf-8',errors='ignore')
			data.insert(row)
