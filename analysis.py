import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


DATA_PATH = "~/data/startupml/ontime_arrival_challenge"


cols_of_interests = ["QUARTER","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME", "ARR_DELAY_NEW"]
independent_vars = ["QUARTER","MONTH","AIRLINE_ID","FL_NUM","ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_STATE_FIPS","ORIGIN_WAC","DEST_AIRPORT_ID","DEST_CITY_MARKET_ID","DEST_STATE_FIPS","DEST_WAC","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME"]
response_var = "ARR_DELAY_NEW"

training_data = pd.read_csv(DATA_PATH+"/training_data.csv")
training_data_x = training_data[independent_vars]
training_data_y = training_data[response_var]


testing_data = pd.read_csv(DATA_PATH+"/testing_data.csv")
testing_data_x = testing_data[independent_vars]
testing_data_y = testing_data[response_var]


regr = linear_model.LinearRegression()
regr.fit(training_data_x.as_matrix(), training_data_y.as_matrix())
print regr.score(testing_data_x, testing_data_y)

# http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
# print('Coefficients: \n', regr.coef_)
# print("Mean squared error: %.2f" % np.mean((regr.predict(testing_data_x) - testing_data_y) ** 2))
# # Plot outputs
# # plt.scatter(testing_data_x, testing_data_y,  color='black')
# plt.plot(testing_data_x, regr.predict(testing_data_x), color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()