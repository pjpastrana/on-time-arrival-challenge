import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt


DATA_PATH = "~/data/startupml/ontime_arrival_challenge"


independent_vars = ["ARR_DELAY","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_STATE_FIPS","DEST_WAC","DEP_TIME","TAXI_OUT","TAXI_IN","ARR_TIME"]
response_var = "DEP_DELAY"
# 

training_data = pd.read_csv(DATA_PATH+"/training_data_2014_2015.csv")
# TODO: run different variables
training_data_x = training_data[independent_vars]
training_data_y = training_data[response_var]


testing_data = pd.read_csv(DATA_PATH+"/testing_data_2016.csv")
testing_data_sample = testing_data[["ARR_DELAY","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_STATE_FIPS","DEST_WAC","DEP_TIME","TAXI_OUT","TAXI_IN","ARR_TIME", "DEP_DELAY"]].dropna()
testing_data_x = testing_data_sample[independent_vars]
testing_data_y = testing_data_sample[response_var]


regr = linear_model.LinearRegression()
regr.fit(training_data_x.as_matrix(), training_data_y.as_matrix())

print regr.score(testing_data_x, testing_data_y)
print mean_squared_error(testing_data_y, regr.predict(testing_data_x))
print("Mean squared error: %.2f" % np.mean((regr.predict(testing_data_x) - testing_data_y) ** 2))

print (regr.predict(testing_data_x) - testing_data_y)

# # Plot outputs
# plt.scatter(testing_data_x["DEP_TIME"], testing_data_y,  color='black')
# plt.plot(testing_data_x["DEP_TIME"], regr.predict(testing_data_x), color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()


#--------------------------------------
# Trying step-wise regression

# stepwise_result = f_regression(training_data_x, training_data_y)
# print stepwise_result 
#--------------------------------------

# Doing a one variable at a time regression

# for var in independent_vars:
#     print var
#     training_data_x = training_data[var]
#     testing_data_x = testing_data[var]
#     regr = linear_model.LinearRegression()
#     regr.fit(training_data_x.as_matrix(), training_data_y.as_matrix())
#     print mean_squared_error(testing_data_y, regr.predict(testing_data_x))
#     training_data_x.drop()
#     testing_data_x.drop()