# 07 · Wind & Solar Energy Production Forecasting (R)

> **MAXD 5133 Applied Statistical Method · Project · group · January 2026**

## 1 · Why this project exists

Renewable energy is not dispatchable. You cannot tell the wind to blow harder at 6pm, so a grid
operator has to **know in advance** roughly how much wind and solar will arrive — otherwise they
either burn gas they did not need or fall short and buy at spot prices.

That makes energy production a forecasting problem with a real cost attached to being wrong.

The project has two halves, both in R, and they answer different questions:

| Task | Question |
|---|---|
| **1 · Forecasting** | given past production, what does next year look like? |
| **2 · Dataset construction** | what do you do when the variable you need only exists as pictures? |

---

## 2 · The concepts behind this project

### 2.1 What makes a time series different

In ordinary regression, rows are assumed independent. In a time series they are emphatically not:
this month's production depends on last month's. That dependence — **autocorrelation** — is the
thing being modelled, and it breaks most of the standard toolkit.

A time series is usually decomposed into three parts:

- **Trend** — the long-run direction (capacity being added year on year).
- **Seasonality** — a repeating pattern with a fixed period (solar peaks in summer, every year).
- **Remainder / noise** — what is left.

The **frequency** is how many observations make one seasonal cycle. For monthly data with an
annual cycle, `frequency = 12`. Setting this wrong is the most common R time-series mistake:
declare it 1 and every seasonal model silently becomes non-seasonal.

### 2.2 Stationarity, and why differencing exists

A series is **stationary** when its statistical properties — mean, variance, autocorrelation —
do not change over time. Most forecasting theory assumes stationarity, and real energy data is
not stationary: it trends upward as capacity grows.

**Differencing** is the fix: model the *change* rather than the level.

```text
first difference:      y'_t = y_t - y_(t-1)          removes a linear trend
seasonal difference:   y'_t = y_t - y_(t-12)         removes an annual cycle
```

The `d` in ARIMA(p,d,q) is how many times this was done. Over-differencing is a real failure mode
— it injects noise and inflates the forecast interval — which is one reason `auto.arima` choosing
`d` by test is safer than picking it by eye.

### 2.3 ARIMA — what the three letters mean

**ARIMA(p, d, q)**, optionally with a seasonal part (P, D, Q)[m]:

| Term | Name | What it says |
|---|---|---|
| **AR(p)** | autoregressive | this month is a weighted sum of the previous *p* months |
| **I(d)** | integrated | the series was differenced *d* times to make it stationary (2.2) |
| **MA(q)** | moving average | this month is a weighted sum of the previous *q* forecast **errors** |

The MA part is the one people misread. It is not a rolling mean of the data — it is a rolling
mean of the *shocks*. If an unusually still month knocked production down, an MA term lets that
shock decay over the following months rather than vanishing instantly.

`auto.arima()` searches over (p,d,q) and the seasonal terms and picks by **AIC** — a criterion
that rewards fit and penalises parameter count, so it prefers the simplest model that explains
the data.

### 2.4 ETS — a completely different idea

**ETS** stands for Error, Trend, Seasonal. Instead of modelling autocorrelation, it models the
three **components** directly and updates each with an exponentially decaying weight, so recent
observations matter more than old ones.

Each component can be **none**, **additive** or **multiplicative**, which gives model families
like ETS(A,N,A) or ETS(M,A,M). Multiplicative seasonality is the right choice when the seasonal
swing grows with the level — which is common in energy, because a 20% summer uplift on a bigger
fleet is a bigger absolute number.

**Why compare ARIMA and ETS rather than picking one:** they encode different assumptions and
therefore **fail differently**. ARIMA is usually stronger when the autocorrelation structure is
rich; ETS is usually more robust when the series is short or the seasonality is clean. If both
agree, you have some confidence. If they disagree, the disagreement itself is information.

### 2.5 Why the split must be chronological

This is the single most important methodological point in the project.

A random train/test split on time-series data lets the model **see the future while predicting
the past**. Hold out a random 20% of months and the model trains on December while forecasting
November — and the reported accuracy is fiction.

The correct procedure is a **chronological holdout**: train on the first 80% of the timeline,
test on the last 20%, and refit the models on the training window only. Re-using a model fitted
on the full data and then "evaluating" it on part of that data is the same mistake wearing a
different hat.

(The more rigorous version is **rolling-origin cross-validation** — forecast one step, roll
forward, repeat — which uses the data better. This project uses the single chronological split,
which is the standard first step.)

### 2.6 Forecast accuracy metrics

| Metric | What it is | When to use it |
|---|---|---|
| **RMSE** | root mean squared error | penalises large misses heavily — use when big errors are disproportionately costly |
| **MAE** | mean absolute error | typical error size, in the series' own units |
| **MAPE** | mean absolute percentage error | scale-free, so series of different sizes can be compared |

MAPE has a trap worth knowing: it is **undefined when the actual value is zero** and it is
asymmetric — it punishes over-forecasting more than under-forecasting. For solar production,
which is genuinely zero overnight, that matters, and it is part of why the analysis works at
monthly rather than hourly resolution.

### 2.7 Time-series-aware imputation

Filling a gap with the column mean is wrong for a time series, and specifically wrong here.

Solar production at 3am is zero. Fill a missing 3am reading with the daily mean and you have
invented electricity generated in the dark — and you have flattened the daily cycle the model is
supposed to learn.

`imputeTS` uses the **structure** instead: interpolation between neighbours, seasonal decomposition,
or Kalman smoothing, so a 3am gap gets filled with what 3am looks like.

The general rule: **the imputation method must respect the structure the data has.** Mean
imputation assumes exchangeable rows, and time-series rows are not exchangeable.

### 2.8 Aggregation level — where the signal lives

Hourly data is noisy, full of overnight zeros, and dominated by the daily cycle. Monthly data is
smooth and shows the annual cycle clearly.

Neither is "right". The rule is to **aggregate to the level where the pattern you care about
lives**: hourly for diagnostics and data-quality checks, monthly for the seasonal forecast. A
12-period seasonality only makes sense once you are on monthly data.

### 2.9 Raster data, and turning pictures into numbers

Task 2 is a different kind of problem. **MODIS MOD11A1** is a NASA product giving daily land
surface temperature as a **raster** — a grid of pixels where each cell holds a measured value, not
a colour.

Two things to know:

- **Scale factors.** Satellite products store values as integers to save space, with a published
  scale factor and offset. Apply them or your temperatures are out by orders of magnitude.
- **Aggregation to a joinable key.** A raster is a grid; a production series is one number per
  date. Reducing the raster (e.g. a spatial mean over the region of interest) is what makes the
  join possible.

---

## 3 · Task 1 — Forecasting energy production

### 3.1 Data preparation, and every check that came with it

```r
library(readr); library(dplyr); library(tidyr); library(lubridate)
library(ggplot2); library(janitor); library(skimr)
library(imputeTS)      # time-series-aware imputation

energy_data <- read_csv("../data/Raw Energy Production Dataset.csv") %>%
  clean_names()        # Start_Hour -> start_hour, consistent snake_case everywhere
```

**Step 1 · One real timestamp.** The raw file stores a date and an hour in separate columns;
nothing time-series works until they are one `POSIXct`.

```r
energy_data <- energy_data %>%
  mutate(
    date       = mdy(date),
    start_hour = as.integer(start_hour),
    datetime   = as.POSIXct(date) + hours(start_hour)
  ) %>%
  arrange(datetime)
```

**Step 2 · The midnight rollover.** `end_hour = 0` after `start_hour = 23` is not an error, it is
the next day. Treat it naively and every day gets one interval of **−23 hours**.

```r
energy_data <- energy_data %>%
  mutate(
    end_hour = as.integer(end_hour),
    end_datetime = case_when(
      end_hour == 0 & start_hour == 23 ~ as.POSIXct(date + days(1)) + hours(0),
      TRUE                             ~ as.POSIXct(date) + hours(end_hour)
    ),
    interval_hours = as.numeric(difftime(end_datetime, datetime, units = "hours"))
  )
```

This is the kind of bug that produces a plausible-looking dataset and a wrong model. Nothing
errors; one interval per day is simply negative, and any duration-weighted aggregate is quietly
corrupted.

**Step 3 · Validate instead of assume.** Three explicit checks: hours outside 0–23, interval
lengths that are not 1 hour, and missing values per column.

```r
invalid_hours  <- energy_data %>% filter(start_hour < 0 | start_hour > 23 |
                                         end_hour  < 0 | end_hour  > 23)
interval_check <- energy_data %>% count(interval_hours) %>% arrange(interval_hours)
missing_values <- colSums(is.na(energy_data))
```

`interval_check` is the one that earns its keep — it is how the rollover bug in Step 2 was found
in the first place. Every interval should be exactly 1; anything else is a data problem you now
know about.

**Step 4 · Duplicates, defined properly.** A duplicate here is the same `datetime` **and** the
same `source` — wind and solar legitimately share a timestamp.

```r
energy_data <- energy_data %>% distinct(datetime, source, .keep_all = TRUE)
```

Deduplicating on `datetime` alone would delete half the dataset, and it would look like it worked.

**Step 5 · Aggregate up** (2.8). Hourly → daily totals → monthly totals.

```r
daily_ts_data <- ts_data %>%
  mutate(date = as.Date(datetime)) %>%
  group_by(date) %>%
  summarise(daily_production = sum(production, na.rm = TRUE)) %>%
  arrange(date)
```

**Step 6 · Look at it before modelling** — a daily trend line and an average-by-month plot with
the months ordered properly (`factor(..., levels = month.name)`, otherwise ggplot sorts April
first, alphabetically, and the seasonal shape is unreadable).

### 3.2 Models

```r
library(forecast); library(tseries)

energy_ts <- ts(monthly_ts_data$monthly_production,
                start = c(start_year, start_month),
                frequency = 12)          # 12 = monthly seasonality

arima_model    <- auto.arima(energy_ts)  # picks p,d,q (+ seasonal) by AIC
arima_forecast <- forecast(arima_model, h = 12)

ets_model    <- ets(energy_ts)           # picks error/trend/season form automatically
ets_forecast <- forecast(ets_model, h = 12)
```

The `frequency = 12` argument is the one that matters (2.1) — it is what tells R there is an
annual cycle to model. `h = 12` forecasts one year ahead.

The two models are the ARIMA and ETS families from 2.3 and 2.4: autocorrelation versus
components, two different sets of assumptions, compared on purpose.

### 3.3 Evaluation done the right way round

```r
train_size <- floor(0.8 * length(energy_ts))
train_ts   <- window(energy_ts, end   = time(energy_ts)[train_size])
test_ts    <- window(energy_ts, start = time(energy_ts)[train_size + 1])

arima_train <- auto.arima(train_ts)      # refit on training only
ets_train   <- ets(train_ts)

arima_accuracy <- accuracy(forecast(arima_train, h = length(test_ts)), test_ts)
ets_accuracy   <- accuracy(forecast(ets_train,   h = length(test_ts)), test_ts)
```

**`window()`, not `sample()`** — this is 2.5 in code. And the models are **refit on the training
window**, not reused from the full-data fit, for the same reason: a model that has already seen
the test months cannot be honestly tested on them.

Accuracy is compared on RMSE / MAE / MAPE (2.6), and the forecasts are overlaid on the actual
test series so the **failure modes are visible**, not just summarised. A model can have a
respectable RMSE while systematically missing every peak, and only the overlay shows that.

---

## 4 · Task 2 — Building a dataset from NASA heat-map images

The second half answers a different question: **what if the variable you need is only available as
pictures?** Temperature drives both solar output and electricity demand, so it is a natural
covariate — but the source is satellite rasters, not a CSV.

```r
library(earthdatalogin); library(terra); library(curl)

edl_netrc()                              # NASA Earthdata authentication for the session
links <- readLines("mod11a1_links.txt")  # the MODIS MOD11A1 granules we need

hdf_dir <- "MOD11A1_HDF"; dir.create(hdf_dir, showWarnings = FALSE)
setwd(hdf_dir)

for (url in links) {
  filename <- basename(url)
  if (!file.exists(filename)) {          # resumable: skip what is already there
    cat("Downloading:", filename, "\n")
    edl_download(url, destfile = filename)
  } else {
    cat("Already exists:", filename, "\n")
  }
}
```

Then **image → numeric** (2.9): each `.hdf` granule is opened with `terra`, the land-surface-
temperature layer is pulled out, the scale factor applied, and the raster reduced to numbers that
can be joined to the production series by date.

Two habits worth keeping from this script:

- **The download loop is idempotent.** Re-running does not re-download 22 granules. Bulk downloads
  fail partway through — that is normal, not exceptional — so the loop is written to be resumed.
- **Credentials come from `.netrc` via `edl_netrc()`**, not typed into the script. A credential in
  source control is a credential that has leaked.

---

## 5 · What I took from this project

1. **Most of forecasting is calendar handling.** The midnight rollover and the duplicate
   definition took longer than fitting both models. That ratio is normal and worth expecting.
2. **Aggregate to the level where the signal lives** (2.8) — hourly for diagnostics, monthly for
   seasonality.
3. **Chronological splits, always**, for anything indexed by time (2.5). This is the mistake that
   most often makes a forecasting result worthless.
4. **Imputation must respect the structure** (2.7): `imputeTS` fills a 3am gap with what 3am looks
   like; a column mean would invent a daily cycle that never happened.
5. **Compare model families, not hyperparameters.** ARIMA and ETS disagreeing tells you something
   about the data. Two ARIMAs disagreeing tells you about AIC.

## 6 · How this connects to the other projects

- The chronological-split argument in 2.5 is exactly the weakness I name in
  [04 · Airbnb](../04-airbnb-price-occupancy-knime), where a random partition was used on
  season-dependent price data. Same mistake, same fix.
- The "validate instead of assume" checks in 3.1 are the same instinct as the duplicate-row count
  in Airbnb and the duplicate-edge bug in
  [06 · Neo4j](../06-neo4j-career-recommender): count things before you model them, because the
  dangerous data bugs do not raise errors.

## Files

```text
R/01_time_series_forecasting_energy.R      cleaning, validation, aggregation, ARIMA, ETS, accuracy
R/02_dataset_from_nasa_heatmap_images.R    authenticated bulk download + image-to-numeric
data/mod11a1_links.txt                     the MODIS granule list
```

The raw energy CSV (2.7 MB) is not in the repo; the script expects it at
`data/Raw Energy Production Dataset.csv`.
