# ==============================
# Question 1
# ==============================

# Observed frequencies
observed <- c(142, 57, 51, 50)

# Expected proportions
expected_prop <- c(0.53, 0.19, 0.14, 0.14)

# Chi-square goodness-of-fit test
chisq.test(observed, p = expected_prop)


# ==============================
# Question 2
# ==============================

# Create the observed frequency table
tornadoes <- matrix(
  c(26, 4, 87, 97,
    2, 41, 46, 63,
    13, 25, 18, 225),
  nrow = 3,
  byrow = TRUE
)

# Add row and column names
rownames(tornadoes) <- c("January", "February", "March")
colnames(tornadoes) <- c("2015", "2014", "2013", "2012")

# Perform Chi-square test of independence
chisq.test(tornadoes)
