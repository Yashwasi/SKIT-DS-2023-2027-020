# Exploratory Data Analysis Findings
## Project: AI-Based Raindrop Volume Prediction System

### 1. Dataset
The initial analysis used 50,000 observations from `RD1_EachDrop[Foliated_Cedar1a_015].csv`.

### 2. Dataset Attributes
- DateTime
- Diameter
- Velocity

### 3. Data Quality
- Missing values in the sample: 0
- Non-positive diameter values: 0
- Non-positive velocity values: 0
- Exact duplicate rows in the sample: 501

### 4. Preliminary Statistics
- Mean diameter: approximately 0.8503
- Mean velocity: approximately 2.8440
- Diameter range: 0.521 to 5.974
- Velocity range: 1.001 to 8.373

Units require confirmation from the original dataset documentation.

### 5. Visualizations
Three visualizations were generated:
1. Raindrop diameter distribution
2. Raindrop diameter versus velocity
3. Distribution of calculated raindrop volume

### 6. Preliminary Conclusion
The sample contains measurable variation in raindrop diameter and velocity. Further investigation is required to validate units, examine unusual measurements, and determine a scientifically appropriate machine-learning prediction target.

### 7. Next Steps
- Verify dataset units and provenance.
- Investigate duplicate records and extreme measurements.
- Define prediction inputs and target.
- Establish training and testing procedures.
- Train and evaluate suitable baseline models.
