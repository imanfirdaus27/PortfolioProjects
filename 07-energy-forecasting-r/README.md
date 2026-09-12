# Wind & Solar Energy Production Forecasting (R)

**Applied Statistical Method (MAXD 5133), Project, UTeM · January 2026 · group**

## Task 1 - Forecasting energy production
Hourly wind and solar production data, cleaned in R (`janitor::clean_names`, date + hour
combined into a `POSIXct` datetime, missing values interpolated with `imputeTS`), then
modelled as a time series: decomposition, **ARIMA** and **exponential smoothing**, compared
on forecast accuracy to pick the better approach. Results were presented in Power BI.

## Task 2 - Building a dataset from NASA heat-map images
The second half turns imagery into numbers: MODIS `MOD11A1` land-surface-temperature images
are downloaded in bulk from a link list, then converted pixel by pixel into a numeric table
that can join the production data - image-to-numeric conversion, done in R.

## Files
```
R/01_time_series_forecasting_energy.R        cleaning, decomposition, ARIMA, ETS, accuracy
R/02_dataset_from_nasa_heatmap_images.R      bulk image download and image-to-numeric
data/mod11a1_links.txt                       the MODIS download list
```
The raw energy production CSV (2.7 MB) is not included; the script expects it at
`data/Raw Energy Production Dataset.csv`.
