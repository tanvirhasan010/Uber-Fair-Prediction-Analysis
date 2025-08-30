# Uber Fares Project Code
# Data Science Capstone Course in HarvardX
# Author: Md Tanvir Hasan
# Date: 19 June 2025
# Install and load required libraries
install.packages("tidyverse")    # Data manipulation (dplyr, ggplot2, etc.)
install.packages("lubridate")    # Datetime handling
install.packages("geosphere")    # Haversine distance calculation
install.packages("glmnet")       # Regularized regression (Ridge/Lasso)
install.packages("caret")        # Machine learning tools
install.packages("modelr")       # Modeling utilities
install.packages("broom")        # Tidy model outputs

# Load libraries
library(tidyverse)   # Includes dplyr, ggplot2, tidyr, etc.
library(lubridate)   # For datetime operations
library(geosphere)   # For distance calculations
library(glmnet)      # For regularized regression
library(caret)       # For train/test splitting & model evaluation
library(modelr)      # For modeling workflows
library(broom)       # For tidy model summaries

# Load required libraries
library(tidyverse)
library(lubridate)
library(geosphere)

# Load dataset
uber_data <- read_csv("uber_data.csv")

# Function to calculate Haversine distance (in miles)
calculate_distance <- function(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon) {
   distHaversine(
      c(pickup_lon, pickup_lat),
      c(dropoff_lon, dropoff_lat)
   ) / 1609.34  # Convert meters to miles
}

# Feature engineering
uber_processed <- uber_data %>%
   mutate(
      # Convert datetime
      pickup_datetime = as_datetime(pickup_datetime),
      hour = hour(pickup_datetime),
      day_of_week = wday(pickup_datetime, label = TRUE),
      month = month(pickup_datetime, label = TRUE),
      
      # Calculate distance
      distance = calculate_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon),
      
      # Rush hour flag (7-9 AM, 4-6 PM)
      is_rush_hour = ifelse((hour >= 7 & hour <= 9) | (hour >= 16 & hour <= 18), 1, 0),
      
      # Borough classification (simplified)
      pickup_borough = case_when(
         pickup_lat >= 40.70 & pickup_lat <= 40.80 & pickup_lon >= -74.02 & pickup_lon <= -73.93 ~ "Manhattan",
         pickup_lat >= 40.60 & pickup_lat <= 40.70 & pickup_lon >= -74.05 & pickup_lon <= -73.85 ~ "Brooklyn",
         TRUE ~ "Other"
      ),
      dropoff_borough = case_when(
         dropoff_lat >= 40.70 & dropoff_lat <= 40.80 & dropoff_lon >= -74.02 & dropoff_lon <= -73.93 ~ "Manhattan",
         dropoff_lat >= 40.60 & dropoff_lat <= 40.70 & dropoff_lon >= -74.05 & dropoff_lon <= -73.85 ~ "Brooklyn",
         TRUE ~ "Other"
      )
   ) %>%
   filter(
      fare_amount > 2.5,  # Minimum fare
      distance > 0.1,     # Minimum distance
      fare_amount < 150,  # Remove extreme outliers
      distance < 50       # Remove unrealistic trips
   )

# Split into train/test sets
set.seed(42)
train_indices <- sample(1:nrow(uber_processed), 0.9 * nrow(uber_processed))
train_data <- uber_processed[train_indices, ]
test_data <- uber_processed[-train_indices, ]

mean_fare <- mean(train_data$fare_amount)
baseline_rmse <- sqrt(mean((test_data$fare_amount - mean_fare)^2))
cat("Baseline RMSE:", baseline_rmse, "\n")

distance_model <- lm(fare_amount ~ distance, data = train_data)
distance_pred <- predict(distance_model, newdata = test_data)
distance_rmse <- sqrt(mean((test_data$fare_amount - distance_pred)^2))
cat("Distance Model RMSE:", distance_rmse, "\n")

time_model <- lm(fare_amount ~ distance + hour + is_rush_hour, data = train_data)
time_pred <- predict(time_model, newdata = test_data)
time_rmse <- sqrt(mean((test_data$fare_amount - time_pred)^2))
cat("Distance + Time Model RMSE:", time_rmse, "\n")

location_model <- lm(fare_amount ~ distance + hour + is_rush_hour + pickup_borough + dropoff_borough, data = train_data)
location_pred <- predict(location_model, newdata = test_data)
location_rmse <- sqrt(mean((test_data$fare_amount - location_pred)^2))
cat("Distance + Time + Location Model RMSE:", location_rmse, "\n")

library(glmnet)

# Prepare data for glmnet
X_train <- model.matrix(~ distance + hour + is_rush_hour + pickup_borough + dropoff_borough, data = train_data)
y_train <- train_data$fare_amount
X_test <- model.matrix(~ distance + hour + is_rush_hour + pickup_borough + dropoff_borough, data = test_data)

# Find optimal lambda via cross-validation
cv_model <- cv.glmnet(X_train, y_train, alpha = 0)  # alpha=0 for Ridge
best_lambda <- cv_model$lambda.min

# Train final model
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)
ridge_pred <- predict(ridge_model, s = best_lambda, newx = X_test)
ridge_rmse <- sqrt(mean((test_data$fare_amount - ridge_pred)^2))
cat("Regularized Model RMSE:", ridge_rmse, "\n")

