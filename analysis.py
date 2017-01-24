import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


DATA_PATH = "~/data/startupml/ontime_arrival_challenge"


# FEB2014 = read_csv(DATA_PATH, "/FEB2014/108043198_T_ONTIME.csv")
# MAR2014 = read_csv(DATA_PATH, "/MAR2014/108043198_T_ONTIME.csv")
# APR2014 = read_csv(DATA_PATH, "/APR2014/108043198_T_ONTIME.csv")
# MAY2014 = read_csv(DATA_PATH, "/MAY2014/108043198_T_ONTIME.csv")
# JUN2014 = read_csv(DATA_PATH, "/JUN2014/108043198_T_ONTIME.csv")
# JUL2014 = read_csv(DATA_PATH, "/JUL2014/108043198_T_ONTIME.csv")
# AUG2014 = read_csv(DATA_PATH, "/AUG2014/108043198_T_ONTIME.csv")
# SEP2014 = read_csv(DATA_PATH, "/SEP2014/108043198_T_ONTIME.csv")
# OCT2014 = read_csv(DATA_PATH, "/OCT2014/108043198_T_ONTIME.csv")
# NOV2014 = read_csv(DATA_PATH, "/NOV2014/108043198_T_ONTIME.csv")
# DEC2014 = read_csv(DATA_PATH, "/DEC2014/108043198_T_ONTIME.csv")

cols_of_interests = ["QUARTER","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME", "ARR_DELAY_NEW"]
independent_vars = ["QUARTER","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME"]
response_var = "ARR_DELAY_NEW"

JAN2014 = pd.read_csv(DATA_PATH+"/JAN2014/108043198_T_ONTIME.csv")
JAN2014 = JAN2014[cols_of_interests]
JAN2014.dropna(inplace=True)
training_data_x = JAN2014[independent_vars]
training_data_y = JAN2014[response_var]


JAN2015 = pd.read_csv(DATA_PATH+"/JAN2015/108043198_T_ONTIME.csv")
JAN2015 = JAN2015[cols_of_interests]
JAN2015.dropna(inplace=True)
testing_data_x = JAN2015[independent_vars]
testing_data_y = JAN2015[response_var]


regr = linear_model.LinearRegression()
regr.fit(training_data_x.as_matrix(), training_data_y.as_matrix())
print regr.score(testing_data_x, testing_data_y)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % np.mean((regr.predict(testing_data_x) - testing_data_y) ** 2))
# Plot outputs
# plt.scatter(testing_data_x, testing_data_y,  color='black')
plt.plot(testing_data_x, regr.predict(testing_data_x), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()