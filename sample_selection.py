import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "~/data/startupml/ontime_arrival_challenge"
cols_of_interests = ["QUARTER","MONTH","AIRLINE_ID","TAIL_NUM","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","ARR_DELAY"]

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

jan2014 = jan2014[cols_of_interests].dropna()
feb2014 = feb2014[cols_of_interests].dropna()
mar2014 = mar2014[cols_of_interests].dropna()
apr2014 = apr2014[cols_of_interests].dropna()
may2014 = may2014[cols_of_interests].dropna()
jun2014 = jun2014[cols_of_interests].dropna()
jul2014 = jul2014[cols_of_interests].dropna()
aug2014 = aug2014[cols_of_interests].dropna()
sep2014 = sep2014[cols_of_interests].dropna()
oct2014 = oct2014[cols_of_interests].dropna()
nov2014 = nov2014[cols_of_interests].dropna()
dec2014 = dec2014[cols_of_interests].dropna()
jan2015 = jan2015[cols_of_interests].dropna()
feb2015 = feb2015[cols_of_interests].dropna()
mar2015 = mar2015[cols_of_interests].dropna()
apr2015 = apr2015[cols_of_interests].dropna()
may2015 = may2015[cols_of_interests].dropna()
jun2015 = jun2015[cols_of_interests].dropna()
jul2015 = jul2015[cols_of_interests].dropna()
aug2015 = aug2015[cols_of_interests].dropna()
sep2015 = sep2015[cols_of_interests].dropna()
oct2015 = oct2015[cols_of_interests].dropna()
nov2015 = nov2015[cols_of_interests].dropna()
dec2015 = dec2015[cols_of_interests].dropna()


training_data = pd.concat([
    jan2014,feb2014,mar2014,apr2014,may2014,jun2014,jul2014,aug2014,sep2014,oct2014,nov2014,dec2014,
    jan2015,feb2015,mar2015,apr2015,may2015,jun2015,jul2015,aug2015,sep2015,oct2015,nov2015,dec2015
])
training_data.to_csv(DATA_PATH+"/training_data_2014_2015.csv", index=False)

#*********************************
# Correlation visualization
#*********************************
corr = training_data.corr()
# plt.matshow(corr )
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# plt.imshow(training_data, cmap='hot', interpolation='nearest')
# plt.show()


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

testing_data = pd.concat([
    jan2016,feb2016,mar2016,apr2016,may2016,jun2016,jul2016,aug2016,sep2016,oct2016,nov2016
])
testing_data.to_csv(DATA_PATH+"/testing_data_2016.csv", index=False)
