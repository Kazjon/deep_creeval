import pymongo, csv, sys
import numpy as np

species = "Turdus_migratorius"
year1 = 2009
year2 = 2012
sample = 10000
outfn = "ebird_sample"


keep_fields = [u'YEAR', u'TIME', u'ELEV_NED', u'DIST_FROM_FLOWING_FRESH', u'LONGITUDE', u'DIST_FROM_WET_VEG_BRACKISH', u'HOUSING_PERCENT_VACANT', u'DIST_FROM_WET_VEG_FRESH', u'DIST_FROM_STANDING_FRESH', u'DIST_FROM_STANDING_BRACKISH', u'CAUS_TEMP_MIN', u'CAUS_TEMP_MAX', u'UMD2011_LANDCOVER', u'CAUS_LAST_SPRING_32F_MEAN', u'CAUS_PREC', u'CAUS_SNOW', u'CAUS_TEMP_AVG', u'CAUS_FIRST_AUTUMN_32F_MEAN', u'POP00_SQMI', u'DIST_FROM_FLOWING_BRACKISH', u'CAUS_LAST_SPRING_32F_MEDIAN', u'CAUS_LAST_SPRING_32F_EXTREME', u'DAY', u'ELEV_GT', u'CAUS_FIRST_AUTUMN_32F_MEDIAN', u'CAUS_FIRST_AUTUMN_32F_EXTREME', u'HOUSING_DENSITY', u'LATITUDE', u'_id']

client = pymongo.MongoClient()
db = client["creeval"]
coll = db["ebird_top10_2008_2012"]

#find all year1 observations
for yr in [year1,year2]:
	cursor = coll.find({"$and":[ {species:{"$exists": True}}, {"YEAR":yr}]})
	results = []
	for b in cursor:
		results.append(b)
	results = np.array(results)

	print results.shape
	print results[0]
	results = results[np.random.choice(len(results),size=sample)]
	print results.shape
	print results[0]

	with open(outfn+"_"+str(yr)+".csv", "wb") as out1:
		writer = csv.DictWriter(out1, fieldnames=keep_fields,extrasaction="ignore")
		writer.writeheader()
		for r in results:
			writer.writerow(r)