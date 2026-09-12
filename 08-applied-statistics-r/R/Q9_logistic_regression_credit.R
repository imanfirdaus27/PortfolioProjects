# ==============================
# Question 9
# ==============================

# Install ISLR package (only need to run once)
install.packages("ISLR")

# Load ISLR package
library(ISLR)

# Load the Credit dataset
credit_data <- Credit

# View structure of dataset
str(credit_data)

# Summary statistics of dataset
summary(credit_data)

#View Data
credit_data

# Convert Married from factor (Yes/No) to numeric (1/0)
# Yes = 1, No = 0
credit_data$Married <- ifelse(credit_data$Married == "Yes", 1, 0)


# a.	Fit the data using logistic regression by taking all variables.
model_full <- glm(
  Married ~ .,              # Use all variables as predictors
  data = credit_data,       # Dataset
  family = binomial         # Logistic regression
)

# Display model summary
summary(model_full)

# b.	Identify the variables that are significant and drop variables that are not significant.

# c.	Fit again with significant variables.
# Fit reduced logistic regression model
model_reduced <- glm(
  Married ~ Rating + Ethnicity,
  data = credit_data,
  family = binomial
)

# Display reduced model summary
summary(model_reduced)

# d.	Split the data into training and testing of your own choice.
# Set seed so results can be reproduced
set.seed(123)

# Create 70% training index
train_index <- sample(
  1:nrow(credit_data),
  0.7 * nrow(credit_data)
)

# Split data
train_data <- credit_data[train_index, ]
test_data  <- credit_data[-train_index, ]



# e.	Predict the testing data and find its accuracy and area under the curve.
#Predict Probabilities
pred_prob <- predict(
  model_reduced,
  test_data,
  type = "response"
)
pred_prob

#Convert to class (0 / 1)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
pred_class

#Accuracy
accuracy <- mean(pred_class == test_data$Married)
accuracy


# AUC
library(pROC)
roc_obj <- roc(test_data$Married, pred_prob)
auc(roc_obj)







