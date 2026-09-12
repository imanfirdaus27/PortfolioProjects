# Install required packages (run once)
install.packages(c("psych", "GPArotation"))

# Load libraries
library(psych)
library(GPArotation)

# Load dataset
cars <- read.csv("../data/cars.csv")

# View structure
str(cars)
summary(cars)

# (a) Identify How Many Factors Should Be Extracted
ev <- eigen(cor(cars))
ev$values

scree(cars)

fa.parallel(cars, fa = "fa", n.iter = 100)



# (b)	Find the correlation matrix.
cor_matrix <- cor(cars)
round(cor_matrix, 2)

# (c) Orthogonal Rotation — VARIMAX
efa_varimax <- fa(
  cars,
  nfactors = 4,
  rotate = "varimax",
  fm = "ml"
)

print(efa_varimax$loadings, cutoff = 0.3)
fa.diagram(efa_varimax, simple = TRUE)

# (d) Oblique Rotation — PROMAX
efa_promax <- fa(
  cars,
  nfactors = 4,     # from part (a)
  rotate = "promax",
  fm = "ml"
)

# Factor loading matrix
print(efa_promax$loadings, cutoff = 0.3)


# (e) — Factor Diagram (Oblique, simple = TRUE)
fa.diagram(efa_promax, simple = TRUE)
