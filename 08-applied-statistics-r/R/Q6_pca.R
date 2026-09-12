# Load dataset
decathlon <- read.csv("../data/decathlon.csv")
decathlon

# Check structure
str(decathlon)

# Store athlete names separately
athlete_name <- decathlon[, 1]

# Select the 10 decathlon event performance variables
decathlon_perf <- decathlon[, 2:11]

# Convert all performance variables to numeric
decathlon_perf <- data.frame(
  lapply(decathlon_perf, function(x) as.numeric(as.character(x)))
)

# Verify structure
str(decathlon_perf)

# Check for NA values
colSums(is.na(decathlon_perf))

# (a) Identify How Many Principal Components Should Be Used
# Run PCA with All Components
pca_full <- prcomp(decathlon_perf, scale. = TRUE)
summary(pca_full)

# Extract Eigenvalues
eigenvalues <- pca_full$sdev^2
eigenvalues

# Scree Plot
plot(eigenvalues,
     type = "b",
     xlab = "Principal Component",
     ylab = "Eigenvalue",
     main = "Scree Plot")
abline(h = 1, col = "red")


# b.	Run the principal component analysis for variable 1 to 10 by using the suggestion in (a).
# PCA using the selected number of components (4 PCs)
pca_b <- prcomp(decathlon_perf, scale. = TRUE)

# PCA summary
summary(pca_b)


# c.	Obtain the principal loadings.
# Obtain principal loadings for the first 4 PCs
loadings <- pca_b$rotation[, 1:4]

# Round for easier interpretation
round(loadings, 3)


# e. Run the principal component analysis for variable 1 to 10 by using 4 factors
# PCA using 4 principal components
pca_4 <- prcomp(decathlon_perf, scale. = TRUE)

# PCA summary
summary(pca_4)

# Principal loadings for first 4 PCs
loadings_4 <- pca_4$rotation[, 1:4]
round(loadings_4, 3)


# g.	Obtain the scores for the one that is better.
# Obtain scores for the better PCA (from part b)
scores_b <- pca_b$x[, 1:4]

# Combine with athlete names for interpretation
pca_scores <- data.frame(
  Athlete = athlete_name,
  PC1 = scores_b[, 1],
  PC2 = scores_b[, 2],
  PC3 = scores_b[, 3],
  PC4 = scores_b[, 4]
)

# View first few scores
head(pca_scores)






