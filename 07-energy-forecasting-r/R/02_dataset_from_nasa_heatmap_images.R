# =========================================================
# Chapter 1: Data Download and Automation
# =========================================================


# Install the authentication package
install.packages("earthdatalogin")
install.packages("terra")  # for reading raster data
install.packages("curl")   # optional, low-level download

# # Load librariesa
library(earthdatalogin)
library(terra)
library(curl)

# Authenticate with NASA Earthdata
# This will prompt you for username/password, or use your environment variables
edl_netrc()  # sets up .netrc and authentication for the session

# Read MODIS download links
# Make sure your text file is in the working directory
links <- readLines("mod11a1_links.txt")
length(links)  # check number of URLs


# Create folder for HDF files
hdf_dir <- "MOD11A1_HDF"
dir.create(hdf_dir, showWarnings = FALSE)

# Change working directory to the folder
setwd(hdf_dir)

# Download each file
for (url in links) {
  filename <- basename(url)  # just the file name, not full path
  
  if (!file.exists(filename)) {  # skip if already downloaded
    cat("Downloading:", filename, "\n")
    edl_download(url, destfile = filename)
  } else {
    cat("Already exists:", filename, "\n")
  }
}


# =========================================================
# Chapter 2: Image-to-Numeric Conversion
# =========================================================

# List all downloaded HDF files
hdf_files <- list.files(pattern = "\\.hdf$")
length(hdf_files)   # should be 22

# Create empty list to store numeric data for each image
lst_all_values <- list()

# Loop through each MODIS heatmap image
for (i in seq_along(hdf_files)) {
  
  # Load heatmap image (Land Surface Temperature - Day)
  lst_raster <- rast(hdf_files[i], subds = "LST_Day_1km")
  
  # (Optional) display heatmap
  plot(
    lst_raster,
    main = paste("MODIS LST Heatmap - Image", i)
  )
  
  # Convert pixel values to numerical data
  lst_values <- values(lst_raster)
  
  # Remove missing values
  lst_values <- na.omit(lst_values)
  
  # Store numeric values in list (one element per image)
  lst_all_values[[i]] <- lst_values
  
  # Show progress
  cat("Image", i, "processed with", length(lst_values), "pixels\n")
}

summary(lst_all_values)


# DISPLAY ALL HEATMAP IMAGES (1–22) AFTER PROCESSING
# Set grid layout (5 x 5 fits 22 images)
par(mfrow = c(5, 5), mar = c(2, 2, 2, 2))

for (i in seq_along(hdf_files)) {
  
  lst_raster <- rast(hdf_files[i], subds = "LST_Day_1km")
  
  plot(
    lst_raster,
    main = paste("Image", i),
    axes = FALSE
  )
}

# Reset plotting layout
par(mfrow = c(1, 1))


# =========================================================
# Chapter 3: Data set Quality and Structure
# =========================================================

# =========================================================
# 3.1 Create a clean, well-structured dataset
# =========================================================

nasa_df <- data.frame(
  image_id = integer(),
  pixel_value = numeric()
)

for (i in seq_along(lst_all_values)) {
  temp_df <- data.frame(
    image_id = i,
    pixel_value = lst_all_values[[i]]
  )
  nasa_df <- rbind(nasa_df, temp_df)
}

names(nasa_df)[2] <- "pixel_value"
str(nasa_df)

library(dplyr)

# =========================================================
# 3.2 Handle missing or invalid values
# =========================================================

# Check total number of pixel values before cleaning
total_pixels_before <- nrow(nasa_df)

# Remove missing values (NA)
nasa_clean <- nasa_df %>%
  filter(!is.na(pixel_value))

# Remove invalid or extreme values (basic rule: must be > 0)
nasa_clean <- nasa_clean %>%
  filter(pixel_value > 0)

# Check total number of pixel values after cleaning
total_pixels_after <- nrow(nasa_clean)

# Display counts
total_pixels_before
total_pixels_after


# =========================================================
# Chapter 4: R Analysis and Visualization
# =========================================================


# Convert MODIS pixel to Celsius (important for report)
# MODIS scale factor = 0.02
# Kelvin → Celsius

nasa_clean$temperature_c <- (nasa_clean$pixel_value * 0.02) - 273.15

summary(nasa_clean$temperature_c)

# Basic Statistical Analysis
mean_temp <- mean(nasa_clean$temperature_c)
median_temp <- median(nasa_clean$temperature_c)
sd_temp <- sd(nasa_clean$temperature_c)

mean_temp
median_temp
sd_temp

# Histogram – Temperature Distribution
hist(
  nasa_clean$temperature_c,
  breaks = 40,
  main = "Distribution of Land Surface Temperature",
  xlab = "Temperature (°C)"
)

# Boxplot – Outlier Detection
boxplot(
  nasa_clean$temperature_c,
  main = "Boxplot of Land Surface Temperature",
  ylab = "Temperature (°C)"
)

#Average Temperature Per Image
avg_by_image <- nasa_clean %>%
  group_by(image_id) %>%
  summarise(avg_temp = mean(temperature_c))

avg_by_image

# Plot trend
plot(
  avg_by_image$image_id,
  avg_by_image$avg_temp,
  type = "b",
  xlab = "MODIS Image Number",
  ylab = "Average Land Surface Temperature (°C)",
  main = "Trend of Average MODIS Land Surface Temperature",
  cex.lab = 1.2,
  cex.main = 1.2
)

grid()

write.csv(nasa_clean,"NASA_LST_clean.csv",row.names=FALSE)
saveRDS(nasa_clean,"NASA_LST_clean.rds")



