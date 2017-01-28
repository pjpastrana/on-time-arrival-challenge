import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

DATA_PATH = "~/data/ontime_arrival_challenge"
# cols_of_interests = ["QUARTER","MONTH","AIRLINE_ID","TAIL_NUM","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","ARR_DELAY"]
cols_of_interests = ["QUARTER","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","DEP_DELAY","DEP_DELAY_NEW","DEP_DEL15","DEP_DELAY_GROUP","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","ARR_DELAY","ARR_DELAY_NEW","ARR_DEL15","ARR_DELAY_GROUP","CANCELLED","DIVERTED","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","FLIGHTS","DISTANCE","DISTANCE_GROUP"]

#******************************
# Read in training data
#******************************

jan2014 = pd.read_csv(DATA_PATH+"/JAN2014/108043198_T_ONTIME.csv")
feb2014 = pd.read_csv(DATA_PATH+"/FEB2014/108043198_T_ONTIME.csv")
mar2014 = pd.read_csv(DATA_PATH+"/MAR2014/108043198_T_ONTIME.csv")
apr2014 = pd.read_csv(DATA_PATH+"/APR2014/108043198_T_ONTIME.csv")
may2014 = pd.read_csv(DATA_PATH+"/MAY2014/108043198_T_ONTIME.csv")
jun2014 = pd.read_csv(DATA_PATH+"/JUN2014/108043198_T_ONTIME.csv")
jul2014 = pd.read_csv(DATA_PATH+"/JUL2014/108043198_T_ONTIME.csv")
aug2014 = pd.read_csv(DATA_PATH+"/AUG2014/108043198_T_ONTIME.csv")
sep2014 = pd.read_csv(DATA_PATH+"/SEP2014/108043198_T_ONTIME.csv")
oct2014 = pd.read_csv(DATA_PATH+"/OCT2014/108043198_T_ONTIME.csv")
nov2014 = pd.read_csv(DATA_PATH+"/NOV2014/108043198_T_ONTIME.csv")
dec2014 = pd.read_csv(DATA_PATH+"/DEC2014/108043198_T_ONTIME.csv")
jan2015 = pd.read_csv(DATA_PATH+"/JAN2015/108043198_T_ONTIME.csv")
feb2015 = pd.read_csv(DATA_PATH+"/FEB2015/108043198_T_ONTIME.csv")
mar2015 = pd.read_csv(DATA_PATH+"/MAR2015/108043198_T_ONTIME.csv")
apr2015 = pd.read_csv(DATA_PATH+"/APR2015/108043198_T_ONTIME.csv")
may2015 = pd.read_csv(DATA_PATH+"/MAY2015/108043198_T_ONTIME.csv")
jun2015 = pd.read_csv(DATA_PATH+"/JUN2015/108043198_T_ONTIME.csv")
jul2015 = pd.read_csv(DATA_PATH+"/JUL2015/108043198_T_ONTIME.csv")
aug2015 = pd.read_csv(DATA_PATH+"/AUG2015/108043198_T_ONTIME.csv")
sep2015 = pd.read_csv(DATA_PATH+"/SEP2015/108043198_T_ONTIME.csv")
oct2015 = pd.read_csv(DATA_PATH+"/OCT2015/108043198_T_ONTIME.csv")
nov2015 = pd.read_csv(DATA_PATH+"/NOV2015/108043198_T_ONTIME.csv")
dec2015 = pd.read_csv(DATA_PATH+"/DEC2015/108043198_T_ONTIME.csv")

#**************************************************
# Convert categorical variable to unique integer
#**************************************************

jan2014["TAIL_NUM"] = jan2014["TAIL_NUM"].astype('category').cat.codes
feb2014["TAIL_NUM"] = feb2014["TAIL_NUM"].astype('category').cat.codes
mar2014["TAIL_NUM"] = mar2014["TAIL_NUM"].astype('category').cat.codes
apr2014["TAIL_NUM"] = apr2014["TAIL_NUM"].astype('category').cat.codes
may2014["TAIL_NUM"] = may2014["TAIL_NUM"].astype('category').cat.codes
jun2014["TAIL_NUM"] = jun2014["TAIL_NUM"].astype('category').cat.codes
jul2014["TAIL_NUM"] = jul2014["TAIL_NUM"].astype('category').cat.codes
aug2014["TAIL_NUM"] = aug2014["TAIL_NUM"].astype('category').cat.codes
sep2014["TAIL_NUM"] = sep2014["TAIL_NUM"].astype('category').cat.codes
oct2014["TAIL_NUM"] = oct2014["TAIL_NUM"].astype('category').cat.codes
nov2014["TAIL_NUM"] = nov2014["TAIL_NUM"].astype('category').cat.codes
dec2014["TAIL_NUM"] = dec2014["TAIL_NUM"].astype('category').cat.codes
jan2015["TAIL_NUM"] = jan2015["TAIL_NUM"].astype('category').cat.codes
feb2015["TAIL_NUM"] = feb2015["TAIL_NUM"].astype('category').cat.codes
mar2015["TAIL_NUM"] = mar2015["TAIL_NUM"].astype('category').cat.codes
apr2015["TAIL_NUM"] = apr2015["TAIL_NUM"].astype('category').cat.codes
may2015["TAIL_NUM"] = may2015["TAIL_NUM"].astype('category').cat.codes
jun2015["TAIL_NUM"] = jun2015["TAIL_NUM"].astype('category').cat.codes
jul2015["TAIL_NUM"] = jul2015["TAIL_NUM"].astype('category').cat.codes
aug2015["TAIL_NUM"] = aug2015["TAIL_NUM"].astype('category').cat.codes
sep2015["TAIL_NUM"] = sep2015["TAIL_NUM"].astype('category').cat.codes
oct2015["TAIL_NUM"] = oct2015["TAIL_NUM"].astype('category').cat.codes
nov2015["TAIL_NUM"] = nov2015["TAIL_NUM"].astype('category').cat.codes
dec2015["TAIL_NUM"] = dec2015["TAIL_NUM"].astype('category').cat.codes

#**************************************************
# Remove cols with duplicated info
# - apply atomic bomb on missing values
#**************************************************

# jan2014 = jan2014[cols_of_interests]#.dropna()
# feb2014 = feb2014[cols_of_interests]#.dropna()
# mar2014 = mar2014[cols_of_interests]#.dropna()
# apr2014 = apr2014[cols_of_interests]#.dropna()
# may2014 = may2014[cols_of_interests]#.dropna()
# jun2014 = jun2014[cols_of_interests]#.dropna()
# jul2014 = jul2014[cols_of_interests]#.dropna()
# aug2014 = aug2014[cols_of_interests]#.dropna()
# sep2014 = sep2014[cols_of_interests]#.dropna()
# oct2014 = oct2014[cols_of_interests]#.dropna()
# nov2014 = nov2014[cols_of_interests]#.dropna()
# dec2014 = dec2014[cols_of_interests]#.dropna()
# jan2015 = jan2015[cols_of_interests]#.dropna()
# feb2015 = feb2015[cols_of_interests]#.dropna()
# mar2015 = mar2015[cols_of_interests]#.dropna()
# apr2015 = apr2015[cols_of_interests]#.dropna()
# may2015 = may2015[cols_of_interests]#.dropna()
# jun2015 = jun2015[cols_of_interests]#.dropna()
# jul2015 = jul2015[cols_of_interests]#.dropna()
# aug2015 = aug2015[cols_of_interests]#.dropna()
# sep2015 = sep2015[cols_of_interests]#.dropna()
# oct2015 = oct2015[cols_of_interests]#.dropna()
# nov2015 = nov2015[cols_of_interests]#.dropna()
# dec2015 = dec2015[cols_of_interests]#.dropna()


training_data = pd.concat([
    jan2014[cols_of_interests],feb2014[cols_of_interests],mar2014[cols_of_interests],apr2014[cols_of_interests],may2014[cols_of_interests],jun2014[cols_of_interests],jul2014[cols_of_interests],aug2014[cols_of_interests],sep2014[cols_of_interests],oct2014[cols_of_interests],nov2014[cols_of_interests],dec2014[cols_of_interests],
    jan2015[cols_of_interests],feb2015[cols_of_interests],mar2015[cols_of_interests],apr2015[cols_of_interests],may2015[cols_of_interests],jun2015[cols_of_interests],jul2015[cols_of_interests],aug2015[cols_of_interests],sep2015[cols_of_interests],oct2015[cols_of_interests],nov2015[cols_of_interests],dec2015[cols_of_interests]
])
len(training_data)
training_data.shape
training_data = training_data.dropna()
len(training_data.dropna())

training_data.to_csv(DATA_PATH+"/training_data_2014_2015.csv", index=False)

#*********************************
# visualization
#*********************************
corr = training_data.corr()
# plt.matshow(corr )
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

total = float(len(training_data))
#print col,sum(np.isnan(training_data[col])) / total
# print col, min(training_data[col]), max(training_data[col])
for col in cols_of_interests:
    try:
        plt.figure()
        h = sorted(training_data[col])
        fit = stats.norm.pdf(h, np.mean(h), np.std(h))
        plt.plot(h,fit,'-o')
        plt.hist(h, normed=True)
        plt.savefig(DATA_PATH+"/distribution_40_cols_norm_fit/"+col+".png")
        plt.close()
    except TypeError:
        continue
    
# feature scaling
scaled_training_data = training_data
# training_data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
scaled_training_data["QUARTER"] = (training_data["QUARTER"] - np.mean(training_data["QUARTER"])) / (np.max(training_data["QUARTER"]) - np.min(training_data["QUARTER"]))
scaled_training_data["MONTH"] = (training_data["MONTH"] - np.mean(training_data["MONTH"])) / (np.max(training_data["MONTH"]) - np.min(training_data["MONTH"]))
scaled_training_data["DAY_OF_MONTH"] = (training_data["DAY_OF_MONTH"] - np.mean(training_data["DAY_OF_MONTH"])) / (np.max(training_data["DAY_OF_MONTH"]) - np.min(training_data["DAY_OF_MONTH"]))
scaled_training_data["DAY_OF_WEEK"] = (training_data["DAY_OF_WEEK"] - np.mean(training_data["DAY_OF_WEEK"])) / (np.max(training_data["DAY_OF_WEEK"]) - np.min(training_data["DAY_OF_WEEK"]))
scaled_training_data["AIRLINE_ID"] = (training_data["AIRLINE_ID"] - np.mean(training_data["AIRLINE_ID"])) / (np.max(training_data["AIRLINE_ID"]) - np.min(training_data["AIRLINE_ID"]))
scaled_training_data["FL_NUM"] = (training_data["FL_NUM"] - np.mean(training_data["FL_NUM"])) / (np.max(training_data["FL_NUM"]) - np.min(training_data["FL_NUM"]))
scaled_training_data["ORIGIN_AIRPORT_ID"] = (training_data["ORIGIN_AIRPORT_ID"] - np.mean(training_data["ORIGIN_AIRPORT_ID"])) / (np.max(training_data["ORIGIN_AIRPORT_ID"]) - np.min(training_data["ORIGIN_AIRPORT_ID"]))
scaled_training_data["ORIGIN_CITY_MARKET_ID"] = (training_data["ORIGIN_CITY_MARKET_ID"] - np.mean(training_data["ORIGIN_CITY_MARKET_ID"])) / (np.max(training_data["ORIGIN_CITY_MARKET_ID"]) - np.min(training_data["ORIGIN_CITY_MARKET_ID"]))
scaled_training_data["ORIGIN_STATE_FIPS"] = (training_data["ORIGIN_STATE_FIPS"] - np.mean(training_data["ORIGIN_STATE_FIPS"])) / (np.max(training_data["ORIGIN_STATE_FIPS"]) - np.min(training_data["ORIGIN_STATE_FIPS"]))
scaled_training_data["ORIGIN_WAC"] = (training_data["ORIGIN_WAC"] - np.mean(training_data["ORIGIN_WAC"])) / (np.max(training_data["ORIGIN_WAC"]) - np.min(training_data["ORIGIN_WAC"]))
scaled_training_data["DEST_AIRPORT_ID"] = (training_data["DEST_AIRPORT_ID"] - np.mean(training_data["DEST_AIRPORT_ID"])) / (np.max(training_data["DEST_AIRPORT_ID"]) - np.min(training_data["DEST_AIRPORT_ID"]))
scaled_training_data["DEST_CITY_MARKET_ID"] = (training_data["DEST_CITY_MARKET_ID"] - np.mean(training_data["DEST_CITY_MARKET_ID"])) / (np.max(training_data["DEST_CITY_MARKET_ID"]) - np.min(training_data["DEST_CITY_MARKET_ID"]))
scaled_training_data["DEST_STATE_FIPS"] = (training_data["DEST_STATE_FIPS"] - np.mean(training_data["DEST_STATE_FIPS"])) / (np.max(training_data["DEST_STATE_FIPS"]) - np.min(training_data["DEST_STATE_FIPS"]))
scaled_training_data["DEST_WAC"] = (training_data["DEST_WAC"] - np.mean(training_data["DEST_WAC"])) / (np.max(training_data["DEST_WAC"]) - np.min(training_data["DEST_WAC"]))
scaled_training_data["CRS_DEP_TIME"] = (training_data["CRS_DEP_TIME"] - np.mean(training_data["CRS_DEP_TIME"])) / (np.max(training_data["CRS_DEP_TIME"]) - np.min(training_data["CRS_DEP_TIME"]))
scaled_training_data["DEP_TIME"] = (training_data["DEP_TIME"] - np.mean(training_data["DEP_TIME"])) / (np.max(training_data["DEP_TIME"]) - np.min(training_data["DEP_TIME"]))
scaled_training_data["DEP_DELAY"] = (training_data["DEP_DELAY"] - np.mean(training_data["DEP_DELAY"])) / (np.max(training_data["DEP_DELAY"]) - np.min(training_data["DEP_DELAY"]))
scaled_training_data["DEP_DELAY_NEW"] = (training_data["DEP_DELAY_NEW"] - np.mean(training_data["DEP_DELAY_NEW"])) / (np.max(training_data["DEP_DELAY_NEW"]) - np.min(training_data["DEP_DELAY_NEW"]))
scaled_training_data["DEP_DEL15"] = (training_data["DEP_DEL15"] - np.mean(training_data["DEP_DEL15"])) / (np.max(training_data["DEP_DEL15"]) - np.min(training_data["DEP_DEL15"]))
scaled_training_data["DEP_DELAY_GROUP"] = (training_data["DEP_DELAY_GROUP"] - np.mean(training_data["DEP_DELAY_GROUP"])) / (np.max(training_data["DEP_DELAY_GROUP"]) - np.min(training_data["DEP_DELAY_GROUP"]))
# scaled_training_data["DEP_TIME_BLK"] = (training_data["DEP_TIME_BLK"] - np.mean(training_data["DEP_TIME_BLK"])) / (np.max(training_data["DEP_TIME_BLK"]) - np.min(training_data["DEP_TIME_BLK"]))
scaled_training_data["TAXI_OUT"] = (training_data["TAXI_OUT"] - np.mean(training_data["TAXI_OUT"])) / (np.max(training_data["TAXI_OUT"]) - np.min(training_data["TAXI_OUT"]))
scaled_training_data["WHEELS_OFF"] = (training_data["WHEELS_OFF"] - np.mean(training_data["WHEELS_OFF"])) / (np.max(training_data["WHEELS_OFF"]) - np.min(training_data["WHEELS_OFF"]))
scaled_training_data["WHEELS_ON"] = (training_data["WHEELS_ON"] - np.mean(training_data["WHEELS_ON"])) / (np.max(training_data["WHEELS_ON"]) - np.min(training_data["WHEELS_ON"]))
scaled_training_data["TAXI_IN"] = (training_data["TAXI_IN"] - np.mean(training_data["TAXI_IN"])) / (np.max(training_data["TAXI_IN"]) - np.min(training_data["TAXI_IN"]))
scaled_training_data["CRS_ARR_TIME"] = (training_data["CRS_ARR_TIME"] - np.mean(training_data["CRS_ARR_TIME"])) / (np.max(training_data["CRS_ARR_TIME"]) - np.min(training_data["CRS_ARR_TIME"]))
scaled_training_data["ARR_TIME"] = (training_data["ARR_TIME"] - np.mean(training_data["ARR_TIME"])) / (np.max(training_data["ARR_TIME"]) - np.min(training_data["ARR_TIME"]))
scaled_training_data["ARR_DELAY"] = (training_data["ARR_DELAY"] - np.mean(training_data["ARR_DELAY"])) / (np.max(training_data["ARR_DELAY"]) - np.min(training_data["ARR_DELAY"]))
scaled_training_data["ARR_DELAY_NEW"] = (training_data["ARR_DELAY_NEW"] - np.mean(training_data["ARR_DELAY_NEW"])) / (np.max(training_data["ARR_DELAY_NEW"]) - np.min(training_data["ARR_DELAY_NEW"]))
scaled_training_data["ARR_DEL15"] = (training_data["ARR_DEL15"] - np.mean(training_data["ARR_DEL15"])) / (np.max(training_data["ARR_DEL15"]) - np.min(training_data["ARR_DEL15"]))
scaled_training_data["ARR_DELAY_GROUP"] = (training_data["ARR_DELAY_GROUP"] - np.mean(training_data["ARR_DELAY_GROUP"])) / (np.max(training_data["ARR_DELAY_GROUP"]) - np.min(training_data["ARR_DELAY_GROUP"]))
# scaled_training_data["ARR_TIME_BLK"] = (training_data["ARR_TIME_BLK"] - np.mean(training_data["ARR_TIME_BLK"])) / (np.max(training_data["ARR_TIME_BLK"]) - np.min(training_data["ARR_TIME_BLK"]))
scaled_training_data["CANCELLED"] = (training_data["CANCELLED"] - np.mean(training_data["CANCELLED"])) / (np.max(training_data["CANCELLED"]) - np.min(training_data["CANCELLED"]))
scaled_training_data["DIVERTED"] = (training_data["DIVERTED"] - np.mean(training_data["DIVERTED"])) / (np.max(training_data["DIVERTED"]) - np.min(training_data["DIVERTED"]))
scaled_training_data["CRS_ELAPSED_TIME"] = (training_data["CRS_ELAPSED_TIME"] - np.mean(training_data["CRS_ELAPSED_TIME"])) / (np.max(training_data["CRS_ELAPSED_TIME"]) - np.min(training_data["CRS_ELAPSED_TIME"]))
scaled_training_data["ACTUAL_ELAPSED_TIME"] = (training_data["ACTUAL_ELAPSED_TIME"] - np.mean(training_data["ACTUAL_ELAPSED_TIME"])) / (np.max(training_data["ACTUAL_ELAPSED_TIME"]) - np.min(training_data["ACTUAL_ELAPSED_TIME"]))
scaled_training_data["AIR_TIME"] = (training_data["AIR_TIME"] - np.mean(training_data["AIR_TIME"])) / (np.max(training_data["AIR_TIME"]) - np.min(training_data["AIR_TIME"]))
scaled_training_data["FLIGHTS"] = (training_data["FLIGHTS"] - np.mean(training_data["FLIGHTS"])) / (np.max(training_data["FLIGHTS"]) - np.min(training_data["FLIGHTS"]))
scaled_training_data["DISTANCE"] = (training_data["DISTANCE"] - np.mean(training_data["DISTANCE"])) / (np.max(training_data["DISTANCE"]) - np.min(training_data["DISTANCE"]))
scaled_training_data["DISTANCE_GROUP"] = (training_data["DISTANCE_GROUP"] - np.mean(training_data["DISTANCE_GROUP"])) / (np.max(training_data["DISTANCE_GROUP"]) - np.min(training_data["DISTANCE_GROUP"]))

arr_delay_log = np.log(training_data["ARR_TIME"])
print sum(np.isnan(arr_delay_log)) / float(len(arr_delay_log))
h = arr_delay_log.dropna()
plt.figure()
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
plt.plot(h,fit,'-o')
plt.hist(h, normed=True)
plt.savefig("ARR_TIME_log.png")
plt.close()

for col in cols_of_interests:
    try:
        plt.figure()
        h = sorted(scaled_training_data[col])
        fit = stats.norm.pdf(h, np.mean(h), np.std(h))
        plt.plot(h,fit,'-o')
        plt.hist(h, normed=True)
        plt.savefig(col+"_scaled.png")
        plt.close()
    except TypeError:
        continue


#----------------------------------------------------------------------------------------------------

#***************************************
# Read in and preprocess testing data
#**************************************

jan2016 = pd.read_csv(DATA_PATH+"/JAN2016/107904530_T_ONTIME.csv")
feb2016 = pd.read_csv(DATA_PATH+"/FEB2016/107904530_T_ONTIME.csv")
mar2016 = pd.read_csv(DATA_PATH+"/MAR2016/107904530_T_ONTIME.csv")
apr2016 = pd.read_csv(DATA_PATH+"/APR2016/107904530_T_ONTIME.csv")
may2016 = pd.read_csv(DATA_PATH+"/MAY2016/107904530_T_ONTIME.csv")
jun2016 = pd.read_csv(DATA_PATH+"/JUN2016/107904530_T_ONTIME.csv")
jul2016 = pd.read_csv(DATA_PATH+"/JUL2016/107904530_T_ONTIME.csv")
aug2016 = pd.read_csv(DATA_PATH+"/AUG2016/107904530_T_ONTIME.csv")
sep2016 = pd.read_csv(DATA_PATH+"/SEP2016/107904530_T_ONTIME.csv")
oct2016 = pd.read_csv(DATA_PATH+"/OCT2016/107904530_T_ONTIME.csv")
nov2016 = pd.read_csv(DATA_PATH+"/NOV2016/107904530_T_ONTIME.csv")

jan2016["TAIL_NUM"] = jan2016["TAIL_NUM"].astype('category').cat.codes
feb2016["TAIL_NUM"] = feb2016["TAIL_NUM"].astype('category').cat.codes
mar2016["TAIL_NUM"] = mar2016["TAIL_NUM"].astype('category').cat.codes
apr2016["TAIL_NUM"] = apr2016["TAIL_NUM"].astype('category').cat.codes
may2016["TAIL_NUM"] = may2016["TAIL_NUM"].astype('category').cat.codes
jun2016["TAIL_NUM"] = jun2016["TAIL_NUM"].astype('category').cat.codes
jul2016["TAIL_NUM"] = jul2016["TAIL_NUM"].astype('category').cat.codes
aug2016["TAIL_NUM"] = aug2016["TAIL_NUM"].astype('category').cat.codes
sep2016["TAIL_NUM"] = sep2016["TAIL_NUM"].astype('category').cat.codes
oct2016["TAIL_NUM"] = oct2016["TAIL_NUM"].astype('category').cat.codes
nov2016["TAIL_NUM"] = nov2016["TAIL_NUM"].astype('category').cat.codes

jan2016 = jan2016[cols_of_interests].dropna()
feb2016 = feb2016[cols_of_interests].dropna()
mar2016 = mar2016[cols_of_interests].dropna()
apr2016 = apr2016[cols_of_interests].dropna()
may2016 = may2016[cols_of_interests].dropna()
jun2016 = jun2016[cols_of_interests].dropna()
jul2016 = jul2016[cols_of_interests].dropna()
aug2016 = aug2016[cols_of_interests].dropna()
sep2016 = sep2016[cols_of_interests].dropna()
oct2016 = oct2016[cols_of_interests].dropna()
nov2016 = nov2016[cols_of_interests].dropna()

# jan2016.dropna(),feb2016.dropna(),mar2016.dropna(),apr2016.dropna(),may2016.dropna(),jun2016.dropna(),jul2016.dropna(),aug2016.dropna(),sep2016.dropna(),oct2016.dropna(),nov2016.dropna()
testing_data = pd.concat([
    jan2016,feb2016,mar2016,apr2016,may2016,jun2016,jul2016,aug2016,sep2016,oct2016,nov2016
])
testing_data.to_csv(DATA_PATH+"/testing_data_2016.csv", index=False)
