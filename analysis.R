library("stringr")

DATA_PATH <- "~/data/startupml/ontime_arrival_challenge"


TRAINING_DATA <- read.csv(str_c(DATA_PATH, "/training_data.csv"))
  
model <- lm(ARR_DELAY_NEW ~ QUARTER+MONTH+AIRLINE_ID+FL_NUM+ORIGIN_AIRPORT_ID+ORIGIN_CITY_MARKET_ID+ORIGIN_STATE_FIPS+ORIGIN_WAC+DEST_AIRPORT_ID+DEST_CITY_MARKET_ID+DEST_STATE_FIPS+DEST_WAC+CRS_DEP_TIME+DEP_TIME+TAXI_OUT+WHEELS_OFF+CRS_ARR_TIME+ARR_TIME, data = TRAINING_DATA)
summary(model)
# predict(model, TESTING_DATA)

# plot(TRAINING_DATA[, cols_of_interest_x], TRAINING_DATA[, cols_of_interest_y])
