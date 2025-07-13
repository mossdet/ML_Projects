# ScalpEEG Brain Heatmaps: Visualization of HFO activity using brain templates

This project uses **K-Means clustering** to segment brain templates and visualize High-Frequency Oscillation (HFO) activity patterns across different age groups using scalp EEG data.

## 🧠 Project Overview

The ScalpEEG Brain Heatmaps project combines computer vision and machine learning clustering to:

1. **Brain Template Segmentation**: Use K-Means clustering to automatically segment brain template images into distinct anatomical regions
2. **HFO Activity Mapping**: Visualize High-Frequency Oscillation occurrence rates across brain regions  
3. **Age Group Analysis**: Compare HFO patterns across developmental age groups (1 month to 17 years)
4. **Interactive Visualization**: Generate brain heatmaps and animated videos showing temporal changes

## 📊 Data Description

The analysis uses anonymized scalp EEG data containing:

- **4,394 records** from pediatric patients
- **Age Groups**: 1month-2yrs, 3-5yrs, 6-10yrs, 11-13yrs, 14-17yrs
- **Brain Regions**: Frontal (F), Parietal (P), Temporal (T), Occipital (O)
- **HFO Occurrence Rate**: Quantified high-frequency oscillation activity per channel

### Data Columns:
- `PatientName`: Anonymized patient identifier
- `Sex`: Gender (1=Male, 2=Female)
- `AgeGroup`: Developmental age category
- `Channel`: EEG electrode location
- `BrainRegion`: Anatomical brain region (F/P/T/O)
- `HFO_OccRate`: High-frequency oscillation occurrence rate

## 🔬 Methodology

### 1. Brain Template Clustering (`clean_brain_with_folds()`)

**Objective**: Remove sulci/folds from brain template images using K-Means clustering

```python
# K-Means clustering for brain cleaning
kmeans = KMeans(n_clusters=4, random_state=42)
```

**Process**:
- Load brain template with anatomical folds (`Brain_With_Folds.png`)
- Resize image to 2000×1300 pixels for processing
- Apply K-Means clustering (n_clusters=4) on RGB pixel values
- Segment image into 4 clusters representing different tissue types
- Generate clean brain template without folds

**Output**: Clean brain template saved as `Clean_Brain_Without_Folds_Template.png`

### 2. Brain Region Parcellation (`get_brain_regions_by_pixel()`)

**Objective**: Segment brain into anatomical lobes using clustering

```python
# K-Means clustering for region parcellation  
kmeans = KMeans(n_clusters=5, random_state=42)
```

**Process**:
- Load colored brain template (`Brain_Lobes.png`)
- Apply K-Means clustering (n_clusters=5) to identify:
  - Background (0)
  - Frontal lobe (1)
  - Parietal lobe (2) 
  - Temporal lobe (3)
  - Occipital lobe (4)
- Apply manual corrections for anatomical accuracy
- Generate parcelled brain template

**Output**: Region-labeled brain template (`Parcelled_Brain_Template.npy`)

### 3. HFO Activity Mapping

**Objective**: Visualize HFO occurrence rates across brain regions and age groups

**Process**:
- Calculate mean HFO occurrence rate per brain region and age group
- Apply viridis colormap for visualization
- Generate heatmaps with two normalization approaches:
  - **Within-group normalization**: Colors relative to each age group's range
  - **Across-group normalization**: Colors relative to all age groups' range
