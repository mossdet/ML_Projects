# 🎨 M5_Clustering - Clustering Techniques

## Overview
This module explores unsupervised learning through clustering algorithms, focusing on discovering hidden patterns and structures in data without labeled examples.

## 📊 Module Structure

## 📁 Projects

### 🧠 ScalpEEG Brain Heatmaps
Generation of Heatmaps using K-Means clustering for brain template segmentation and HFO activity visualization across developmental age groups.
**Key Features:**
- **Brain Template Segmentation**: K-Means clustering (k=4) for automatic removal of anatomical folds
- **Region Parcellation**: K-Means clustering (k=5) for anatomical lobe definition
- **HFO Activity Mapping**: Visualization of neural oscillation patterns across brain regions
- **Developmental Analysis**: Comparative study across 5 age groups (1 month - 17 years)
- **Interactive Visualization**: Animated heatmaps showing temporal changes
<img src="ScalpEEG_Brain_Heatmaps/Output/Norm_Across_Age_Groups/B_1_3to5yrs_brain_regions_cmap_normalized_across_age_groups.png" alt="HFO Brain Heatmap Example" width="600">

### 📊 Module_5_Project_1: Customer Segmentation

### 🧬 Module_5_Project_2: CLuster crimess according to type and the neighborhood

### 📈 Module_5_Project_3: Cancer data clustering

### 🎯 Centroid-Based
- **K-Means**: Partition data into k clusters
- **K-Medoids**: Robust centroid-based clustering
- **Mini-Batch K-Means**: Scalable K-means variant

### 🌊 Density-Based
- **DBSCAN**: Density-based spatial clustering
- **Mean-Shift**: Mode-seeking algorithm
- **OPTICS**: Ordering points clustering

### 📊 Hierarchical
- **Agglomerative**: Bottom-up hierarchy building
- **Divisive**: Top-down cluster splitting
- **Dendrogram Analysis**: Tree-based visualization

### 🧠 Advanced Methods
- **Gaussian Mixture Models**: Probabilistic clustering
- **Spectral Clustering**: Graph-based clustering
- **Affinity Propagation**: Exemplar-based clustering

## 📊 Cluster Evaluation Metrics

- **Silhouette Score**: Cluster cohesion and separation
- **Inertia**: Within-cluster sum of squares
- **Davies-Bouldin Index**: Cluster validity measure
- **Calinski-Harabasz Index**: Cluster separation ratio
- **Adjusted Rand Index**: Clustering agreement measure
- **Homogeneity & Completeness**: Cluster purity measures
- **V-Measure**: Balance between homogeneity and completeness