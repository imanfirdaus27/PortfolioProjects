# Airbnb Price & Occupancy Prediction - Singapore

**Fundamental of Data Science (MAXD 5113), UTeM · November 2025 · group of 3**
Anis, Syahmi and Muhammad Iman Firdaus.

## Question
What actually drives an Airbnb listing's price, and how does price relate to how full the
listing stays? Hosts leaning on Smart Pricing have no view of that trade-off.

## Data
Inside Airbnb Singapore snapshots: **3,693 listings** (79 columns) and **3,881,748 calendar
rows** (Dec 24, Mar 25, Jun 25).
Cleaning: 712,565 duplicate calendar rows removed, price strings (`$120.00`) and percentages
(`95%`) converted to numerics, review-score columns with heavy missingness dropped, outliers
trimmed per room type by IQR, then listings joined to calendar and a **weekly occupancy rate**
engineered as the target.

## Model
XGBoost (Gradient Boosted Trees) built in **KNIME**: File Reader, Statistics, Table
Partitioner, Gradient Boosted Trees Learner/Predictor (one pair for price, one for
occupancy), Numeric Scorer, Rule Engine, then scatter, bar and box visuals and an Excel
writer for the deployment output.

## Results
| Target | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Nightly price | **0.907** | $24.57 | $41.83 | 16.2% |
| Weekly occupancy rate | **0.828** | 0.119 | 0.17 | n/a (zeros in actuals) |

Mean signed difference was ~0 for both, so the models are not systematically over- or
under-predicting, and predicted vs actual held up neighbourhood by neighbourhood. The scatter
of predicted price against predicted occupancy separates a high-utilisation budget segment
($50-150) from a low-utilisation premium segment.

## Files
```
workflow/Airbnb Price and Occupancy Optimization (XGBoost).knwf   KNIME workflow
```
Open with KNIME Analytics Platform (File > Import KNIME Workflow). The Inside Airbnb
snapshots are not included - download them from [insideairbnb.com](http://insideairbnb.com/get-the-data/).

## Figures

The 13 images below are the real figures from the MAXD 5153 Assignment 1 report, extracted from the submitted PDF.

![The full KNIME workflow: reader → cleaning → features → partition → two learner/predictor branches → scorers.](figures/01-knime-workflow.png)

*The full KNIME workflow: reader → cleaning → features → partition → two learner/predictor branches → scorers.*

![Correlation heat map. Price and occupancy are only weakly related — the finding the report hangs on.](figures/02-correlation-heatmap.png)

*Correlation heat map. Price and occupancy are only weakly related — the finding the report hangs on.*

![Occupancy vs price: no clean downward slope.](figures/03-occupancy-vs-price.png)

*Occupancy vs price: no clean downward slope.*

![Price distribution by neighbourhood.](figures/04-price-by-neighbourhood.png)

*Price distribution by neighbourhood.*

![Top 10 neighbourhoods by occupancy — not the same list as by price.](figures/05-top10-neighbourhoods-occupancy.png)

*Top 10 neighbourhoods by occupancy — not the same list as by price.*

![Listings by lat/long, coloured by price.](figures/06-geospatial-price.png)

*Listings by lat/long, coloured by price.*

![Same map coloured by occupancy — compare with the previous figure.](figures/07-geospatial-occupancy.png)

*Same map coloured by occupancy — compare with the previous figure.*

![Price vs occupancy split by room type.](figures/08-price-vs-occupancy-by-roomtype.png)

*Price vs occupancy split by room type.*

![Gradient boosting: each tree fits the residual left by the ones before it.](figures/09-gradient-boosting-diagram.png)

*Gradient boosting: each tree fits the residual left by the ones before it.*

![Numeric scorer for the price model.](figures/10-price-model-scorer.png)

*Numeric scorer for the price model.*

![Predicted vs actual price — the model under-predicts expensive listings.](figures/11-price-predicted-vs-actual.png)

*Predicted vs actual price — the model under-predicts expensive listings.*

![Numeric scorer for the occupancy model. Clearly the weaker of the two.](figures/12-occupancy-model-scorer.png)

*Numeric scorer for the occupancy model. Clearly the weaker of the two.*

![Predicted price vs occupancy — pricing up does not cost occupancy the way host forums assume.](figures/13-predicted-price-vs-occupancy.png)

*Predicted price vs occupancy — pricing up does not cost occupancy the way host forums assume.*
