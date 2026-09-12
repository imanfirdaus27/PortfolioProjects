# ==============================
# Chapter 3 Data Preparation and Analysis
# ==============================

# 3.2.1 Define necessary libraries for data cleaning
library(readr)      # read CSV files
library(dplyr)      # data manipulation
library(tidyr)      # tidying data
library(lubridate)  # date-time handling
library(ggplot2)    # visualization
library(janitor)    # clean column names
library(skimr)      # quick data summary

# Optional (if handle missing values later)
library(imputeTS)   # time series missing value imputation (e.g., interpolation)

# 3.2.2 Load data set and clean column names

# Load data set
energy_data <- read_csv("../data/Raw Energy Production Dataset.csv")

# Clean column names to a consistent format
energy_data <- energy_data %>%
  clean_names()

# View first few rows
head(energy_data)

# 3.2.3 Convert Date and Hour into datetime

energy_data <- energy_data %>%
  mutate(
    date = mdy(date),                       # convert Date to Date format
    start_hour = as.integer(start_hour),    # ensure Start_Hour is numeric
    datetime = as.POSIXct(date) + hours(start_hour)
  )

# Sort data by time
energy_data <- energy_data %>%
  arrange(datetime)

# View structure
str(energy_data$datetime)

# 3.2.4 Handle End_Hour and validate hourly consistency

energy_data <- energy_data %>%
  mutate(
    end_hour = as.integer(end_hour),
    
    # Create end_datetime:
    # - Normal case: same date + end_hour
    # - Midnight rollover case: end_hour = 0 after 23 -> next day 00:00
    end_datetime = case_when(
      end_hour == 0 & start_hour == 23 ~ as.POSIXct(date + days(1)) + hours(0),
      TRUE ~ as.POSIXct(date) + hours(end_hour)
    ),
    
    # Expected duration in hours (should be 1 for hourly intervals)
    interval_hours = as.numeric(difftime(end_datetime, datetime, units = "hours"))
  )

# 1) Validate hour ranges
invalid_hours <- energy_data %>%
  filter(start_hour < 0 | start_hour > 23 | end_hour < 0 | end_hour > 23)

nrow(invalid_hours)

# 2) Check interval consistency (expect most to be 1 hour)
interval_check <- energy_data %>%
  count(interval_hours) %>%
  arrange(interval_hours)

interval_check

# 3.2.5 Check missing values

# Count missing values per column
missing_values <- colSums(is.na(energy_data))
missing_values

# Check duplicate records based on datetime and source
duplicate_records <- energy_data %>%
  count(datetime, source) %>%
  filter(n > 1)

nrow(duplicate_records)

# 3.2.6 Remove duplicate records

# Remove duplicates based on datetime and source
energy_data <- energy_data %>%
  distinct(datetime, source, .keep_all = TRUE)

# Re-check duplicates to confirm removal
duplicate_records_after <- energy_data %>%
  count(datetime, source) %>%
  filter(n > 1)

nrow(duplicate_records_after)

# 3.2.7 Select target variable and prepare time series dataset

# Select only required columns for time series analysis
ts_data <- energy_data %>%
  select(datetime, production) %>%
  arrange(datetime)

# Verify structure
str(ts_data)

# Preview data
head(ts_data)

# 3.2.8 Aggregate hourly data into daily totals

daily_ts_data <- ts_data %>%
  mutate(date = as.Date(datetime)) %>%
  group_by(date) %>%
  summarise(
    daily_production = sum(production, na.rm = TRUE)
  ) %>%
  arrange(date)

# Check structure
str(daily_ts_data)

# Preview first rows
head(daily_ts_data)

# 3.3.1 Time Series Trend Analysis (Daily Production)

ggplot(daily_ts_data, aes(x = date, y = daily_production)) +
  geom_line(color = "steelblue") +
  labs(
    title = "Daily Energy Production Over Time",
    x = "Date",
    y = "Daily Energy Production"
  ) +
  theme_minimal()

# 3.3.2 Seasonal Pattern Analysis (Monthly)

monthly_ts_data <- daily_ts_data %>%
  mutate(month = format(date, "%m"),
         month_name = format(date, "%B")) %>%
  group_by(month_name) %>%
  summarise(
    avg_monthly_production = mean(daily_production, na.rm = TRUE)
  )

# Ensure correct month order
monthly_ts_data$month_name <- factor(
  monthly_ts_data$month_name,
  levels = month.name
)

# Plot monthly seasonal pattern
ggplot(monthly_ts_data, aes(x = month_name, y = avg_monthly_production)) +
  geom_line(group = 1, color = "steelblue") +
  geom_point(color = "steelblue") +
  labs(
    title = "Average Daily Energy Production by Month",
    x = "Month",
    y = "Average Daily Energy Production"
  ) +
  theme_minimal()




# # ==============================
# Chapter 4 Model Application
# ================================

# 4.1.1 Load necessary libraries for time series modelling
library(forecast)
library(tseries)
library(lubridate)


# 4.1.2 Prepare Monthly Time Series Data
# (Using daily_ts_data from Chapter 3)
monthly_ts_data <- daily_ts_data %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  summarise(
    monthly_production = sum(daily_production, na.rm = TRUE)
  ) %>%
  arrange(month)

# Check structure (for verification)
str(monthly_ts_data)
head(monthly_ts_data)


# 4.1.3 Convert Monthly Data into Time Series Object
start_date <- min(monthly_ts_data$month)

start_year  <- year(start_date)
start_month <- month(start_date)

energy_ts <- ts(
  monthly_ts_data$monthly_production,
  start = c(start_year, start_month),
  frequency = 12
)

# 4.1.4 Plot time series
autoplot(energy_ts) +
  labs(
    title = "Monthly Energy Production Time Series",
    x = "Year",
    y = "Energy Production"
  ) +
  theme_minimal()



# 4.2 Model 1: ARIMA
arima_model <- auto.arima(energy_ts)
summary(arima_model)


# 4.2.1 ARIMA Forecast (Next 12 Months)
arima_forecast <- forecast(arima_model, h = 12)

autoplot(arima_forecast) +
  labs(
    title = "ARIMA Forecast of Monthly Energy Production",
    x = "Year",
    y = "Energy Production"
  ) +
  theme_minimal()


# 4.3 Model 2: Exponential Smoothing (ETS)
ets_model <- ets(energy_ts)
summary(ets_model)


# 4.3.1 ETS Forecast (Next 12 Months)
ets_forecast <- forecast(ets_model, h = 12)

autoplot(ets_forecast) +
  labs(
    title = "ETS Forecast of Monthly Energy Production",
    x = "Year",
    y = "Energy Production"
  ) +
  theme_minimal()


# # ====================================
# Chapter 5 Model Performance Evaluation
# ======================================

# 5.1 Split data into training (80%) and testing (20%)
train_size <- floor(0.8 * length(energy_ts))

train_ts <- window(energy_ts, end = time(energy_ts)[train_size])
test_ts <- window(energy_ts, start = time(energy_ts)[train_size + 1])

# 5.2 Refit models on training data
arima_train <- auto.arima(train_ts)
ets_train <- ets(train_ts)

# 5.3 Forecast on test data
arima_test_forecast <- forecast(arima_train, h = length(test_ts))
ets_test_forecast <- forecast(ets_train, h = length(test_ts))

# 5.4 Accuracy metrics calculation
arima_accuracy <- accuracy(arima_test_forecast, test_ts)
ets_accuracy <- accuracy(ets_test_forecast, test_ts)

arima_accuracy
ets_accuracy


# 5.5 Visual comparison of forecast vs actual
autoplot(test_ts, series = "Actual") +
  autolayer(arima_test_forecast$mean, series = "ARIMA Forecast") +
  autolayer(ets_test_forecast$mean, series = "ETS Forecast") +
  labs(
    title = "Forecast Comparison: Actual vs Predicted Energy Production",
    x = "Year",
    y = "Energy Production"
  ) +
  theme_minimal()
