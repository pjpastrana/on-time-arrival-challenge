import pandas as pd

# https://www.checkmarket.com/sample-size-calculator/
# population: 450,000
# margin of error: 1%
# confidence level: 95%
# sample size: 9403

DATA_PATH = "~/data/startupml/ontime_arrival_challenge"
cols_of_interests = ["QUARTER","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME", "ARR_DELAY_NEW"]

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

jan2014 = jan2014[cols_of_interests].dropna().sample(n=9400)
feb2014 = feb2014[cols_of_interests].dropna().sample(n=9400)
mar2014 = mar2014[cols_of_interests].dropna().sample(n=9400)
apr2014 = apr2014[cols_of_interests].dropna().sample(n=9400)
may2014 = may2014[cols_of_interests].dropna().sample(n=9400)
jun2014 = jun2014[cols_of_interests].dropna().sample(n=9400)
jul2014 = jul2014[cols_of_interests].dropna().sample(n=9400)
aug2014 = aug2014[cols_of_interests].dropna().sample(n=9400)
sep2014 = sep2014[cols_of_interests].dropna().sample(n=9400)
oct2014 = oct2014[cols_of_interests].dropna().sample(n=9400)
nov2014 = nov2014[cols_of_interests].dropna().sample(n=9400)
dec2014 = dec2014[cols_of_interests].dropna().sample(n=9400)

training_data = pd.concat([jan2014,feb2014,mar2014,apr2014,may2014,jun2014,jul2014,aug2014,sep2014,oct2014,nov2014,dec2014])
training_data.to_csv(DATA_PATH+"/training_data.csv", index=False)

#----------------------------------------------------------------------------------------------------
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

jan2015 = jan2015[cols_of_interests].dropna().sample(n=9400)
feb2015 = feb2015[cols_of_interests].dropna().sample(n=9400)
mar2015 = mar2015[cols_of_interests].dropna().sample(n=9400)
apr2015 = apr2015[cols_of_interests].dropna().sample(n=9400)
may2015 = may2015[cols_of_interests].dropna().sample(n=9400)
jun2015 = jun2015[cols_of_interests].dropna().sample(n=9400)
jul2015 = jul2015[cols_of_interests].dropna().sample(n=9400)
aug2015 = aug2015[cols_of_interests].dropna().sample(n=9400)
sep2015 = sep2015[cols_of_interests].dropna().sample(n=9400)
oct2015 = oct2015[cols_of_interests].dropna().sample(n=9400)
nov2015 = nov2015[cols_of_interests].dropna().sample(n=9400)
dec2015 = dec2015[cols_of_interests].dropna().sample(n=9400)

testing_data = pd.concat([jan2015,feb2015,mar2015,apr2015,may2015,jun2015,jul2015,aug2015,sep2015,oct2015,nov2015,dec2015])
testing_data.to_csv(DATA_PATH+"/testing_data.csv", index=False)