# Superstore Sales & Profit Dashboards (Power BI)

**Big Data Analytics and Visualization (MAXD 5153), Project 1, UTeM · April 2026 · individual**

## What it shows
- **Heat map** (matrix visual): region x product category by sum of sales, with conditional
  formatting so intensity carries the value. Highest: **West / Technology at 42,000**.
  Lowest: **South / Furniture at 12,000**.
- **Profit map by state**: bubble size by profit. **California +76,381** is the strongest
  market; **Texas -25,729** loses money, which is the finding that matters - high sales there
  are not high profit.
- **Product demand bar chart** and a **sales trend line** showing steady growth 2015-2017.

## Reading it
Technology sells across every region, so it deserves more investment; Furniture in the South
underperforms. The gap between California (highest sales and profit) and Texas (negative
profit) says the problem is discounting and cost, not demand.

## Files
```
Superstore Sales Dashboard.pbix          open with Power BI Desktop
data/heat_map_power_bi_dataset.csv       the region x category source table
```

## Figures

The 3 images below are the real figures from the MAXD 5153 Project 1 report, extracted from the submitted PDF.

![Sales heat map, region × category. West 93,500 · East 72,500 · Central 61,500 · South 52,000.](figures/01-region-category-heatmap.png)

*Sales heat map, region × category. West 93,500 · East 72,500 · Central 61,500 · South 52,000.*

![The dashboard page: KPI cards, then trend and breakdown visuals.](figures/02-dashboard.png)

*The dashboard page: KPI cards, then trend and breakdown visuals.*

![Full dashboard with slicers applied, as demoed.](figures/03-dashboard-full.png)

*Full dashboard with slicers applied, as demoed.*
