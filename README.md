# on-time-arrival-challenge

I worked the on-time arrival data challenge.

- Downloaded datasets for all the months for years 2014, 2015, 2016.
- The main hypothesis was that given data from two previous year, we can predict the "current" year.
- After an initial exploration, most of variables with text values were removed, simply because they provided duplicate information already represented in unique id's.
- A new variable was created called ARR/DEP_DELTA to represent the difference between expected departure/arrival and actual departure/arrival. This variable later proved to be not useful based a correlation analysis.
- As part of the exploration, I wanted to know if it was "safe" to remove all missing values. The percentage of missing values was calculated and based on the it low value, missing values were removed.
- A normality test was performed on the data under the assumption that it "had" to be normal for linear regression. That data does not follow a normal distribution transforming the data did not provide any significant improvements on the model.
- Finally based on the correlation analysis, the most significant features were selected for the model.

In conclusion, not using departure delay as input feature limits the scope of prediction, since other features that dont have a strong linear relationship have to be used. That forced the selection of WHEELS_ON as function of WHEELS_OFF, DEP_TIME, ARR_TIME. Under the assumption that once the time at which the plane landed, the arrival delay can be calculated from the expected arrival time. 
Future steps would involve expanding the dataset with interactio between the feature to perform a polynomial regression.