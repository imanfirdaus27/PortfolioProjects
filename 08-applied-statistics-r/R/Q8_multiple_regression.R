# Load dataset
transport <- read.csv("../data/transport_dataset.csv")

# View structure
str(transport)

# View first rows
head(transport)

# a.	Fit a multiple regression model predicting travel time. What is the regression equation?
# Fit multiple regression model
model <- lm(Travel_Time ~ ., data = transport)

# Model summary
summary(model)

# c. Find the confidence interval for the independent variables.
# 95% confidence intervals for regression coefficients
confint(model)


# d.	Check the adequacy of the model. What can you conclude?
# Diagnostic plots
par(mfrow = c(2, 2))
plot(model)


# e. Do the data contain outlier?
# Studentized residuals
stud_res <- rstudent(model)

# Identify potential outliers
which(abs(stud_res) > 3)

stud_res[abs(stud_res) > 3]


# f.	Check the unusual observations. What can you conclude?
# Cook's Distance
cooks <- cooks.distance(model)

# Threshold for influence
threshold <- 4 / nrow(transport)

# Identify influential observations
which(cooks > threshold)

# See the values
cooks[cooks > threshold]

# Leverage values
lev <- hatvalues(model)

# High leverage points (rule of thumb: > 2p/n)
p <- length(coef(model))
which(lev > (2 * p / nrow(transport)))



# g.	Obtain the correlation matrix (only for numeric variables). Paste the matrix into your report. 
# Describe the relationship between dependent variable and each of the predictors (only for numeric variables).
# Select numeric variables only
numeric_transport <- transport[, sapply(transport, is.numeric)]

# Correlation matrix
cor_matrix <- round(cor(numeric_transport), 3)

# Display matrix
cor_matrix




