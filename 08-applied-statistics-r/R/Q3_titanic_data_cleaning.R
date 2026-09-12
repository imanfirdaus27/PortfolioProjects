# ===============================
# Question 3: Titanic Data Cleaning Script
# ===============================

# Load required libraries
install.packages("VIM")
install.packages("mice")

library(dplyr)
library(ggplot2)
library(stringr)
library(VIM)
library(mice)

# Load dataset
titanic <- read.csv(
  "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
  stringsAsFactors = FALSE
)



# 1.	Familiarize yourself with the dataset

# Basic inspection
str(titanic)
dim(titanic)
head(titanic)
tail(titanic)
summary(titanic)

# Check PassengerId uniqueness
any(duplicated(titanic$PassengerId))

# Examine sample Name entries
head(titanic$Name, 10)

# Examine sample Ticket entries
head(titanic$Ticket, 10)


# Descriptive Statistics
# Survival rate overall
prop.table(table(titanic$Survived))

# Survival by sex
prop.table(table(titanic$Survived, titanic$Sex), 2)

# Survival by class
prop.table(table(titanic$Survived, titanic$Pclass), 2)

# Age & Fare summaries
summary(titanic$Age)
summary(titanic$Fare)

# Initial Visualizations
ggplot(titanic, aes(Age)) + geom_histogram(binwidth = 5)
ggplot(titanic, aes(Fare)) + geom_histogram(binwidth = 10)




# 2. Check for structural errors and inconsistencies

# Check duplicate PassengerId
any(duplicated(titanic$PassengerId))

# Check duplicate passenger names
any(duplicated(titanic$Name))



# Validate Value Ranges
# Survived (should be only 0 or 1)
unique(titanic$Survived)

# Pclass (should be only 1, 2, 3)
unique(titanic$Pclass)

# Age (should be between 0 and 80, no negatives)
summary(titanic$Age)
titanic %>% filter(Age < 0 | Age > 80)

# SibSp and Parch (non-negative)
summary(titanic$SibSp)
summary(titanic$Parch)

titanic %>% filter(SibSp < 0 | Parch < 0)

# Fare (non-negative)
summary(titanic$Fare)
titanic %>% filter(Fare < 0)

# Sex (only "male" / "female")
unique(titanic$Sex)

# Embarked (only C / Q / S or NA)
unique(titanic$Embarked)


# Identify Logical Inconsistencies
# Passengers with Age = 0
titanic %>% filter(Age == 0)

# Passengers with Fare = 0
titanic %>% filter(Fare == 0)

# Unusual family combinations
# Very large families
titanic %>% filter(SibSp + Parch > 7)



# Outlier Detection (IQR Method)
# Outliers in Fare
Q1_fare <- quantile(titanic$Fare, 0.25, na.rm = TRUE)
Q3_fare <- quantile(titanic$Fare, 0.75, na.rm = TRUE)
IQR_fare <- Q3_fare - Q1_fare

fare_outliers <- titanic %>%
  filter(Fare < (Q1_fare - 1.5 * IQR_fare) |
           Fare > (Q3_fare + 1.5 * IQR_fare))

nrow(fare_outliers)

# Outliers in Age
Q1_age <- quantile(titanic$Age, 0.25, na.rm = TRUE)
Q3_age <- quantile(titanic$Age, 0.75, na.rm = TRUE)
IQR_age <- Q3_age - Q1_age

age_outliers <- titanic %>%
  filter(Age < (Q1_age - 1.5 * IQR_age) |
           Age > (Q3_age + 1.5 * IQR_age))

nrow(age_outliers)



# Boxplots for Visual Outlier Detection
# Boxplot for Fare
ggplot(titanic, aes(y = Fare)) +
  geom_boxplot() +
  labs(title = "Boxplot of Fare")

# Boxplot for Age
ggplot(titanic, aes(y = Age)) +
  geom_boxplot() +
  labs(title = "Boxplot of Age")


# check Name & Ticket Consistency
# Check Name format
head(titanic$Name, 10)

# Check Ticket format
head(titanic$Ticket, 10)




# 3. Check and Handle Data Type Errors

# Convert Variables to Correct Data Types
# Survived → Factor (No / Yes)
titanic$Survived <- factor(
  titanic$Survived,
  levels = c(0, 1),
  labels = c("No", "Yes")
)

# Pclass → Ordered Factor
titanic$Pclass <- factor(
  titanic$Pclass,
  levels = c(3, 2, 1),
  labels = c("3rd", "2nd", "1st"),
  ordered = TRUE
)

# Sex → Factor
titanic$Sex <- factor(titanic$Sex)

# Embarked → Factor with Meaningful Labels
titanic$Embarked <- factor(
  titanic$Embarked,
  levels = c("C", "Q", "S"),
  labels = c("Cherbourg", "Queenstown", "Southampton")
)


# Verify Numeric and Integer Variables
# Check data types
str(titanic$Age)
str(titanic$Fare)
str(titanic$SibSp)
str(titanic$Parch)


# Trim Whitespace from Text Fields
titanic$Name   <- trimws(titanic$Name)
titanic$Ticket <- trimws(titanic$Ticket)
titanic$Cabin  <- trimws(titanic$Cabin)

# Check for Special Characters / Encoding Issues
# Check for non-ASCII characters in Name
any(grepl("[^[:print:]]", titanic$Name))

# Check for non-ASCII characters in Ticket
any(grepl("[^[:print:]]", titanic$Ticket))



# Verify Final Structure After Conversion
str(titanic)




# 4. Develop and Implement a Missing Data Strategy
titanic$Cabin[titanic$Cabin == ""] <- NA

# Percentage of missing values for each variable
missing_pct <- sapply(titanic, function(x) {
  mean(is.na(x)) * 100
})

round(missing_pct, 2)



# Missing Data Visualization
# Visualize missing data pattern
VIM::aggr(
  titanic,
  numbers = TRUE,
  sortVars = TRUE,
  labels = names(titanic),
  cex.axis = 0.7,
  gap = 3
)



# Missingness Pattern Analysis (Age)
# Check if Age missingness relates to other variables
titanic$Age_Missing <- is.na(titanic$Age)

table(titanic$Age_Missing, titanic$Pclass)
table(titanic$Age_Missing, titanic$Survived)

# Compare Fare by Age missingness
tapply(titanic$Fare, titanic$Age_Missing, median, na.rm = TRUE)



# Age Imputation + Flag Variable

# Save original Age
titanic$Age_Original <- titanic$Age

# Create Age_Imputed flag
titanic$Age_Imputed <- is.na(titanic$Age)

# Create Age_Imputed using median imputation by Pclass and Sex
titanic <- titanic %>%
  group_by(Pclass, Sex) %>%
  mutate(
    Age_Imputed = ifelse(
      is.na(Age_Original),
      median(Age_Original, na.rm = TRUE),
      Age_Original
    )
  ) %>%
  ungroup()




# Impute missing Embarked with mode ("Southampton")
titanic$Embarked[is.na(titanic$Embarked)] <- "Southampton"



# Create Has_Cabin indicator
titanic$Has_Cabin <- ifelse(is.na(titanic$Cabin) | titanic$Cabin == "", 0, 1)

# Extract Cabin Deck letter
titanic$Cabin_Deck <- ifelse(
  titanic$Has_Cabin == 1,
  substr(titanic$Cabin, 1, 1),
  NA
)



# Before vs After Comparison
# Missing Value Counts
# Missing values after cleaning
sapply(titanic, function(x) sum(is.na(x)))

# Age distribution after imputation
ggplot(titanic, aes(Age)) +
  geom_histogram(binwidth = 5, fill = "steelblue") +
  ggtitle("Age Distribution After Imputation")

# Summary Statistics Check
summary(titanic)
titanic




# 5. Create derived variables and enhance the dataset
# Family Variables
# Family size
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1

# Family category
titanic$FamilyCategory <- cut(
  titanic$FamilySize,
  breaks = c(0, 1, 3, 5, Inf),
  labels = c("Alone", "Small", "Medium", "Large"),
  right = TRUE
)

# IsAlone indicator
titanic$IsAlone <- ifelse(titanic$FamilySize == 1, 1, 0)



# Extract Title from Name (text between comma and period)
titanic$Title <- str_trim(
  str_extract(titanic$Name, "(?<=,)[^\\.]+")
)

titanic$Title_Grouped <- dplyr::case_when(
  titanic$Title == "Mr" ~ "Adult Males",
  titanic$Title == "Mrs" ~ "Married Females",
  titanic$Title %in% c("Miss", "Mlle", "Ms") ~ "Unmarried Females",
  titanic$Title == "Master" ~ "Boys",
  titanic$Title %in% c("Capt", "Col", "Major") ~ "Officer",
  titanic$Title %in% c("Dr", "Rev") ~ "Professional",
  titanic$Title %in% c("Sir", "Lady", "Countess") ~ "Noble",
  TRUE ~ "Rare"
)

titanic$Title_Grouped <- factor(titanic$Title_Grouped)



# Age Categories
# Age groups (using imputed Age)
titanic$AgeGroup <- cut(
  titanic$Age,
  breaks = c(-Inf, 12, 19, 59, Inf),
  labels = c("Child", "Teen", "Adult", "Senior")
)

# IsChild indicator
titanic$IsChild <- ifelse(titanic$Age < 18, 1, 0)


# Fare Variables
# Fare per person
titanic$FarePerPerson <- titanic$Fare / titanic$FamilySize

# Fare quartiles
fare_quartiles <- quantile(titanic$FarePerPerson, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)

titanic$FareBracket <- cut(
  titanic$FarePerPerson,
  breaks = c(-Inf, fare_quartiles, Inf),
  labels = c("Low", "Medium-Low", "Medium-High", "High")
)





# 6.	Validate data quality and document the cleaning process

levels(titanic$Survived)
levels(titanic$Pclass)
levels(titanic$Sex)
levels(titanic$Embarked)
levels(titanic$FamilyCategory)
levels(titanic$AgeGroup)
levels(titanic$Title_Grouped)
levels(titanic$FareBracket)

summary(titanic$Age)
summary(titanic$Fare)

any(titanic$Age < 0)
any(titanic$Fare < 0)


all(titanic$FamilySize == titanic$SibSp + titanic$Parch + 1)

sapply(titanic, function(x) sum(is.na(x)))



# Age: Before vs After Imputation
library(tidyr)

age_long <- titanic %>%
  select(Age_Original, Age) %>%
  pivot_longer(cols = c(Age_Original, Age),
               names_to = "Version",
               values_to = "Age")

ggplot(age_long, aes(Age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  facet_wrap(~ Version, ncol = 2) +
  ggtitle("Age Distribution: Before vs After Imputation")


# Fare Distribution (unchanged)
ggplot(titanic, aes(Fare)) +
  geom_histogram(binwidth = 10, fill = "grey70", color = "white") +
  ggtitle("Fare Distribution After Cleaning")



# Verify Survival Rates Remain Unchanged
prop.table(table(titanic$Survived))
# Create survival proportion data
survival_prop <- prop.table(table(titanic$Survived))

# Convert to data frame
survival_df <- as.data.frame(survival_prop)
colnames(survival_df) <- c("Survived", "Proportion")

# Plot
ggplot(survival_df, aes(x = Survived, y = Proportion, fill = Survived)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Survival Proportion of Titanic Passengers",
    y = "Proportion",
    x = "Survival Status"
  ) +
  theme_minimal()



# Correlation Matrix (Numeric Variables)
numeric_vars <- titanic %>%
  select(where(is.numeric))

cor(numeric_vars, use = "complete.obs")



# Cross-Tabulations for Expected Survival Patterns
# Survived vs Pclass
prop.table(table(titanic$Survived, titanic$Pclass), 2)

ggplot(titanic, aes(Pclass, fill = Survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Survival Rate by Passenger Class",
       y = "Proportion")


# Survived vs Sex
prop.table(table(titanic$Survived, titanic$Sex), 2)

ggplot(titanic, aes(Sex, fill = Survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Survival Rate by Sex",
       y = "Proportion")


# Survived vs AgeGroup
prop.table(table(titanic$Survived, titanic$AgeGroup), 2)

ggplot(titanic, aes(AgeGroup, fill = Survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Survival Rate by Age Group",
       y = "Proportion")


# Survived vs FamilyCategory
prop.table(table(titanic$Survived, titanic$FamilyCategory), 2)

ggplot(titanic, aes(FamilyCategory, fill = Survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Survival Rate by Family Category",
       y = "Proportion")


# Save cleaned dataset to CSV
write.csv(
  titanic,
  file = "titanic_cleaned.csv",
  row.names = FALSE
)


