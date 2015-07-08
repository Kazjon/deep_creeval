import pymongo, argparse, csv, sys
from copy import copy
from pymongo import MongoClient

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Use this to import data for use by creeval')
	parser.add_argument('-i','--input',help="Input csv file",required=True)
	parser.add_argument('-o','--output',help='Output collection in DB',required=True)
	parser.add_argument('-t','--timefield',help='Field to use for temporal ordering',required=True)
	parser.add_argument('-y','--fields_y',help='Field(s) to use as y in prediction, all others are assumed to be in X',required=False, default="")
	parser.add_argument('-g','--ignore',help='Field(s) not to import',required=False, default="")
	args = parser.parse_args()
	client = MongoClient()
	db = client.creeval
	datasets = db.datasets
	data = db[args.output]

	with open(args.input) as csvfile:
		reader = csv.DictReader(csvfile)
		fields_x = copy(reader.fieldnames)
		for k,field in enumerate(fields_x):
			fields_x[k] = field.replace(" ","_")
		if not args.timefield in fields_x:
			sys.exit("The timefield provided was not found in the file.  Field names were: "+str(reader.fieldnames))
		for field in args.ignore.split():
			fields_x.remove(field)
		fields_y = args.fields_y.split()
		for field in fields_y:
			if field not in fields_x:
				sys.exit("Field in provided 'fields_y' not found in CSV.")
			fields_x.remove(field)

		datasets.insert({"fields_x": fields_x,"fields_y": fields_y,"name": args.output, "timefield": args.timefield})
		for row in reader:
			for k,v in row.iteritems():
				try:
					vf = float(v)
				except ValueError:
					vf = v
				if " " in k:
					row[k.replace(" ","_")] = vf
					del row[k]
				else:
					row[k] = vf
			for ignore in args.ignore.split():
				del row[ignore]
 			print row
			#print unicode(row,'utf-8',errors='ignore')
			#row['name'] = unicode(row['name'],'utf-8',errors='ignore')
			data.insert(row)
