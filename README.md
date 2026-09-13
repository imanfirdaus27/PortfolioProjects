# Portfolio Projects

**Muhammad Iman Firdaus Bin Md Rostan** — Data Analyst, Workforce & HR Analytics.

This repo holds two sets of work: my **Master of Technology (Data Science & Analytics)**
projects at **Universiti Teknikal Malaysia Melaka (UTeM)**, in the numbered folders below,
and some **earlier analytics work** kept at the root of the repo.

## Master's coursework and research

| # | Project | Course | Stack | Headline result |
|---|---------|--------|-------|-----------------|
| [01](01-flood-segmentation-kd) | Cross-modal knowledge distillation for flood segmentation (**Master's thesis**) | MAXU 5214 | PyTorch, U-Net | SAR baseline 0.6244 IoU vs optical teacher 0.8261 IoU, modality gap 0.2017 |
| [02](02-mvtec-industrial-defect-ml) | Industrial defect ML system on MVTec AD | MAXD 5143 | PySpark, PyTorch, DQN | Pipeline 8.45s to 4.28s (1.97x on 4 workers); INT8 + 30% pruning at 0.9 pp accuracy cost |
| [03](03-ecommerce-purchase-prediction) | E-commerce purchase prediction & customer segmentation | MAXD 5143 | Scikit-learn | Random Forest 89.98% accuracy, F1 63.30% on 12,330 sessions |
| [04](04-airbnb-price-occupancy-knime) | Airbnb price & occupancy prediction, Singapore | MAXD 5113 | KNIME, XGBoost | Price R² 0.907 (MAE $24.57), weekly occupancy R² 0.828 |
| [05](05-imdb-sentiment-nlp) | IMDB movie-review sentiment analysis | MAXD 5153 | TF-IDF, LR/NB/SVM | Logistic Regression ~88% accuracy on 50,000 reviews |
| [06](06-neo4j-career-recommender) | Student career recommendation system | MAXD 5123 | Neo4j, Cypher | Graph model of students, courses, skills and jobs with 30 recommendation queries |
| [07](07-energy-forecasting-r) | Wind & solar energy production forecasting | MAXD 5133 | R, ARIMA, ETS | Forecasting models compared on hourly production data |
| [08](08-applied-statistics-r) | Applied statistical methods in R | MAXD 5133 | R | Chi-square, ANOVA, PCA, factor analysis, regression, logistic regression |
| [09](09-superstore-powerbi) | Superstore sales & profit dashboards | MAXD 5153 | Power BI | Region x category heat map, state profit map, sales trend |
| [10](10-matlab-image-processing-gui) | Interactive image-processing GUI | MAXD 5153 | MATLAB | Load, process and compare images in one app |
| [11](11-hadoop-big-data-architecture) | Hadoop ecosystem & big data architecture (written) | MAXD 5123 | — | Hadoop 1 → YARN → Hadoop 3 → Spark → cloud, and the pain each one solved |

## How this repo is organised

Every folder is self-contained and its `README.md` is a full walkthrough, not a summary. Each one
runs in the same order:

1. **Why this project exists** — the real problem, not the assignment title
2. **The concepts behind it** — the theory explained before it is used, so the rest makes sense
3. **The data**, and the defects found in it before modelling
4. **Method, step by step**, each step with the reason for it and the figure at the point it is needed
5. **Results**, including the ones that did not work
6. **What I would do differently**, and what I would say about it in an interview
7. **How this connects to the other projects** — the same ideas recur across them

The code is exactly as submitted (notebooks are kept, and their code cells are also extracted into
plain `.py` files so the logic is readable without opening Jupyter). The same text is mirrored in
my Notion project notes, so the two never drift.

## A note on group work
Several projects were done in groups; each README names the group and says which part is
mine. Raw university report files are not published here.

## Figures
Where a project had a submitted report, its real figures — plots, dashboards, scorer
screenshots, workflow diagrams — are in that project's `figures/` folder (36 in total).

## Data
Small datasets are included. Large ones (MVTec AD, Sen1Floods11, the Airbnb Singapore
snapshot, hourly energy production) are not - each README links to the source.

## Report figures

65 figures were extracted from the submitted PDF reports and committed alongside the code,
so every project README shows the real charts, dashboards and screenshots rather than
describing them.

| Project | Figures |
|---|---|
| `02-mvtec-industrial-defect-ml/figures/` | 12 |
| `03-ecommerce-purchase-prediction/figures/` | 16 |
| `04-airbnb-price-occupancy-knime/figures/` | 13 |
| `05-imdb-sentiment-nlp/figures/` | 8 |
| `06-neo4j-career-recommender/figures/` | 13 |
| `09-superstore-powerbi/figures/` | 3 |

The same figures are embedded in the Notion project notes, loaded from this repo's `main` branch, so the two stay in sync.

## Earlier work (kept at the repo root)

| File / folder | What it is |
|---|---|
| `Airbnb_Data_Understanding_&_Preparation.ipynb` | Airbnb data understanding and preparation notebook |
| `COVID PORTFOLIO PROJECT SCRIPTS.sql` | COVID data exploration in SQL |
| `NASHVILLE CLEANING DATA PROJECT.sql` | Nashville housing data cleaning in SQL |
| `A Comparative Analysis of Restaurants Across ...` | Undergraduate comparative analysis |
| `Anti Drone Operator System` | Undergraduate system project |
| `Car Rental Payment Method System (Capstone)` | Undergraduate capstone |

---

[LinkedIn](https://www.linkedin.com/in/firdausrostan/) · [GitHub](https://github.com/imanfirdaus27)
