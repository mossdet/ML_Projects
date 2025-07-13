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

## 📈 Key Results

### Age-Related HFO Patterns (Norm_Across_Age_Groups)

The project generates brain heatmaps showing HFO activity patterns across five developmental stages:

1. **1 month - 2 years**: Early developmental HFO patterns
2. **3 - 5 years**: Preschool age neural activity  
3. **6 - 10 years**: School age development
4. **11 - 13 years**: Early adolescent changes
5. **14 - 17 years**: Late adolescent maturation

### Visualization Features

- **Brain Heatmaps**: Color-coded visualization of HFO activity intensity
- **Bar Charts**: Quantitative comparison across age groups and brain regions
- **Animated Videos**: Time-lapse showing developmental changes
- **Multiple Normalization**: Within-group vs. across-group comparisons

## 🚀 Usage

### Prerequisites

```bash
pip install matplotlib opencv-python numpy pandas seaborn scikit-learn
```

### Running the Analysis

```python
# Import and run the complete pipeline
from brain_heatmaps import *

# Execute all analysis steps
if __name__ == "__main__":
    clean_brain_with_folds()           # Step 1: Clean brain template
    get_brain_regions_by_pixel()       # Step 2: Parcellate brain regions  
    get_hfo_heatmaps_per_age_group()   # Step 3: Generate within-group heatmaps
    get_hfo_heatmaps_across_age_groups() # Step 4: Generate across-group heatmaps
    get_hfo_heatmaps_across_sex_groups() # Step 5: Generate sex-based heatmaps
    make_video_within_age_groups()      # Step 6: Create within-group video
    make_video_across_age_groups()      # Step 7: Create across-group video
```

### Individual Function Usage

```python
# Run specific analysis components
clean_brain_with_folds()                    # Brain template cleaning
get_brain_regions_by_pixel()               # Brain parcellation  
get_hfo_heatmaps_across_age_groups()       # Cross-age comparison
```

## 📁 Project Structure

```
ScalpEEG_Brain_Heatmaps/
├── README.md
├── src/
│   └── brain_heatmaps.py              # Main analysis script
├── Data/
│   └── anonymized_data.csv            # HFO dataset
├── Brain_Template/
│   ├── Brain_With_Folds.png           # Original brain template
│   ├── Brain_Lobes.png                # Colored lobe template
│   ├── Clean_Brain_Without_Folds_Template.png  # Processed template
│   └── Parcelled_Brain_Template.npy   # Region-labeled template
└── Output/
    ├── Norm_Across_Age_Groups/         # Cross-age heatmaps
    ├── Norm_Within_Group/              # Within-age heatmaps  
    ├── Norm_Across_Sex_Groups/         # Sex-based heatmaps
    ├── HFO_OccRate_Barchart_Across_Age_Groups.png
    └── Brain_Heatmap_*.avi             # Generated videos
```

## 🔧 Technical Details

### Clustering Parameters
- **Brain Cleaning**: K-Means with k=4 clusters for tissue segmentation
- **Region Parcellation**: K-Means with k=5 clusters for anatomical regions
- **Random State**: 42 (for reproducibility)

### Image Processing
- **Resolution**: 2000×1300 pixels for high-quality analysis
- **Color Space**: RGB for clustering, grayscale for overlays
- **Interpolation**: Linear interpolation for resizing

### Visualization
- **Colormap**: Viridis (default) or customizable
- **Transparency**: Alpha blending for template overlays
- **Output Formats**: PNG images, AVI videos

## 📊 Scientific Applications

This methodology enables researchers to:

1. **Study Brain Development**: Track HFO changes across developmental stages
2. **Identify Biomarkers**: Detect abnormal HFO patterns in neurological conditions
3. **Anatomical Mapping**: Automatically segment brain regions from templates
4. **Temporal Analysis**: Visualize changes in neural activity over time

## 🏥 Clinical Relevance

High-Frequency Oscillations (HFOs) are important biomarkers for:
- **Epilepsy**: Pathological HFOs indicate seizure onset zones
- **Brain Development**: Normal HFOs reflect healthy neural maturation  
- **Cognitive Function**: HFO patterns correlate with cognitive abilities
- **Treatment Monitoring**: Track therapeutic intervention effects

## 🔬 Clustering Innovation

The novel application of K-Means clustering for brain template processing:
- **Automated Segmentation**: Reduces manual annotation requirements
- **Reproducible Results**: Consistent clustering across analyses
- **Scalable Processing**: Efficient for large-scale neuroimaging studies
- **Flexible Framework**: Adaptable to different brain templates and datasets

## 📖 References

This project demonstrates the integration of:
- **Machine Learning**: K-Means clustering for image segmentation
- **Computer Vision**: OpenCV for image processing
- **Neuroimaging**: Brain template analysis and visualization
- **Data Science**: Statistical analysis and visualization of HFO patterns

---

**Note**: This analysis uses anonymized clinical data in compliance with privacy regulations. All patient identifiers have been removed or replaced with anonymous codes.
