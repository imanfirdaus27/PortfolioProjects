# Interactive Image Processing GUI (MATLAB)

**Big Data Analytics and Visualization (MAXD 5153), Assignment 2, UTeM · May 2026 · individual**

A MATLAB `uifigure` application for loading and processing images without writing code.

## Features
- Load one or many images at once (`uigetfile` with `MultiSelect`), browse between them
- Images normalised to a consistent working size on load
- Standard image-processing operations applied to the current image, with the original kept
  so every change can be compared and reverted
- Status feedback in the app while files load

## Files
```
ImageProcessingGUIApplication.m    the full app in one file
```
Run with `ImageProcessingGUIApplication` in MATLAB (Image Processing Toolbox).
