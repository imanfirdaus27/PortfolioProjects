# Load data from URL
poisons <- read.csv("https://raw.githubusercontent.com/guru99-edu/R-Programming/master/poisons.csv")
poisons

# View structure
str(poisons)

# Convert poison and treat to factors
poisons$poison <- factor(poisons$poison)
poisons$treat  <- factor(poisons$treat)


# Step 1: Check the format of the variable poison
# Check structure
str(poisons$poison)

# Check levels
levels(poisons$poison)


# Step 2: Print the summary statistic: count, mean and standard deviation
# Load dplyr
library(dplyr)

# Summary statistics by poison
summary_stats <- poisons %>%
  group_by(poison) %>%
  summarise(
    count = n(),
    mean  = mean(time),
    sd    = sd(time)
  )

summary_stats


# Step 3: Plot a box plot
# Boxplot of survival time by poison
boxplot(time ~ poison,
        data = poisons,
        col = "lightblue",
        xlab = "Poison Type",
        ylab = "Survival Time",
        main = "Survival Time by Poison Type")



# Step 4: Compute the one-way ANOVA test
# One-way ANOVA
anova_poison <- aov(time ~ poison, data = poisons)

# ANOVA summary
summary(anova_poison)


# Step 5: Run a pairwise t-test
# Pairwise t-tests with Bonferroni correction
pairwise.t.test(poisons$time,
                poisons$poison,
                p.adjust.method = "bonferroni")


# Add Treat variable:
# Two-way ANOVA with interaction
anova_two_way <- aov(time ~ poison * treat, data = poisons)

# Summary of two-way ANOVA
summary(anova_two_way)


