"""
ScalpEEG Brain Heatmaps

This module provides a framework for analyzing High-Frequency Oscillations (HFO) 
in scalp EEG data using K-Means clustering for brain template segmentation and visualization.

Classes:
    BrainHeatmapAnalyzer: Main class for brain template processing and HFO analysis
    
Author: Daniel Lachner Piza
E-Mail: dalapiz@proton.me
Date: July 2025
"""

import matplotlib as mpl
mpl.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
import pandas as pd
import os
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import KMeans
from matplotlib import cm


class BrainHeatmapAnalyzer:
    """
    A class for analyzing HFO patterns in scalp EEG data using clustering-based 
    brain template segmentation and visualization.
    
    This class provides methods for:
    - Brain template cleaning and segmentation using K-Means clustering
    - Anatomical region parcellation 
    - HFO activity mapping across age groups and brain regions
    - Generation of heatmaps and animated visualizations
    
    Attributes:
        data_path (str): Path to the EEG data directory
        brain_template_path (str): Path to brain template images
        output_path (str): Path for saving analysis outputs
        img_resize_width (int): Target width for image resizing (default: 2000)
        img_resize_height (int): Target height for image resizing (default: 1300)
        colormap: Matplotlib colormap for visualizations (default: viridis)
        age_groups (List[str]): Age group categories for analysis
        brain_regions (List[str]): Brain region abbreviations (F, P, T, O)
    """
    
    def __init__(self, 
                 data_path: str = "/home/dlp/Development/ML_Projects/M5_Clustering/ScalpEEG_Brain_Heatmaps/Data",
                 brain_template_path: str = "/home/dlp/Development/ML_Projects/M5_Clustering/ScalpEEG_Brain_Heatmaps/Brain_Template",
                 output_path: str = "/home/dlp/Development/ML_Projects/M5_Clustering/ScalpEEG_Brain_Heatmaps/Output",
                 img_resize_width: int = 2000,
                 img_resize_height: int = 1300,
                 colormap = mpl.cm.viridis):
        """
        Initialize the BrainHeatmapAnalyzer with configuration parameters.
        
        Args:
            data_path (str): Directory containing the EEG dataset
            brain_template_path (str): Directory containing brain template images
            output_path (str): Directory for saving analysis results
            img_resize_width (int): Target width for image processing
            img_resize_height (int): Target height for image processing
            colormap: Matplotlib colormap for heatmap visualization
        """
        self.data_path = data_path
        self.brain_template_path = brain_template_path
        self.output_path = output_path
        self.img_resize_width = img_resize_width
        self.img_resize_height = img_resize_height
        self.colormap = colormap
        
        # Analysis parameters
        self.age_groups = ['1monto2yrs', '3to5yrs', '6to10yrs', '11to13yrs', '14to17yrs']
        self.brain_regions = ['F', 'P', 'T', 'O']  # Frontal, Parietal, Temporal, Occipital
        
        # Create necessary directories
        self._create_directories()
        
        # Load data
        self.df = None
        self._load_data()
    
    def _create_directories(self) -> None:
        """Create necessary output directories if they don't exist."""
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.brain_template_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/Norm_Within_Group", exist_ok=True)
        os.makedirs(f"{self.output_path}/Norm_Across_Age_Groups", exist_ok=True)
        os.makedirs(f"{self.output_path}/Norm_Across_Sex_Groups", exist_ok=True)
    
    def _load_data(self) -> None:
        """Load the HFO dataset from CSV file."""
        try:
            data_file = f"{self.data_path}/anonymized_data.csv"
            self.df = pd.read_csv(data_file)
            print(f"Successfully loaded dataset with {len(self.df)} records")
        except FileNotFoundError:
            print(f"Warning: Data file not found at {data_file}")
            self.df = None
    
    def _load_and_resize_image(self, image_path: str) -> np.ndarray:
        """
        Load and resize an image for processing.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Resized RGB image array
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        brain_img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        brain_img_rsz = cv2.resize(brain_img, (self.img_resize_width, self.img_resize_height), 
                                   interpolation=cv2.INTER_LINEAR)
        return brain_img_rsz
    
    def _prepare_image_for_clustering(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image data for K-Means clustering by reshaping RGB values.
        
        Args:
            image (np.ndarray): Input RGB image
            
        Returns:
            np.ndarray: Flattened RGB pixel values ready for clustering
        """
        r, g, b = cv2.split(image)
        bgr_img_for_cluster = np.transpose(np.vstack((r.ravel(), g.ravel(), b.ravel())))
        return bgr_img_for_cluster
    
    def _print_cluster_statistics(self, labels: np.ndarray, n_clusters: int) -> None:
        """
        Print the percentage distribution of pixels across clusters.
        
        Args:
            labels (np.ndarray): Cluster labels from K-Means
            n_clusters (int): Number of clusters
        """
        total_pixels = len(labels)
        for cluster_id in range(n_clusters):
            percentage = np.sum(labels == cluster_id) / total_pixels * 100
            print(f"Cluster {cluster_id} percentage: {percentage:.2f}%")
    
    def clean_brain_with_folds(self) -> None:
        """
        Clean brain template by removing folds/sulci using K-Means clustering.
        
        This method applies K-Means clustering (k=4) to segment the brain template image
        into different tissue types, then creates a cleaned version without anatomical folds.
        The cleaned template is saved for use in subsequent analysis steps.
        
        Output:
            - Clean_Brain_Without_Folds_Template.npy: Numpy array of cleaned brain template
            - Clean_Brain_Without_Folds_Template.png: PNG image of cleaned brain template
        """
        print("Cleaning brain with folds...")
        
        # Load and resize brain template with folds
        img_path = f"{self.brain_template_path}/Brain_With_Folds.png"
        brain_img_rsz = self._load_and_resize_image(img_path)
        
        # Prepare data for clustering
        bgr_img_for_cluster = self._prepare_image_for_clustering(brain_img_rsz)
        
        # Apply K-Means clustering (k=4 for tissue segmentation)
        kmeans = KMeans(n_clusters=4, random_state=42).fit(bgr_img_for_cluster)
        cluster_labels = kmeans.labels_
        
        # Print cluster statistics
        self._print_cluster_statistics(cluster_labels, 4)
        
        # Create cleaned image by assigning grayscale values to clusters
        clean_image = np.zeros_like(cluster_labels)
        clean_image[cluster_labels == 0] = 255  # Background
        clean_image[cluster_labels == 1] = 220  # Brain tissue
        clean_image[cluster_labels == 2] = 128  # Intermediate tissue
        clean_image[cluster_labels == 3] = 220  # Brain tissue
        clean_image = clean_image.reshape(brain_img_rsz.shape[0], brain_img_rsz.shape[1])
        
        # Save cleaned brain template
        np_template_path = f"{self.output_path}/Clean_Brain_Without_Folds_Template.npy"
        np.save(np_template_path, clean_image)
        
        # Save as PNG image
        plt.figure(figsize=(12, 8))
        plt.imshow(clean_image, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.gca().set_facecolor('w')
        plt.savefig(f"{self.output_path}/Clean_Brain_Without_Folds_Template.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        print("Brain template cleaning completed successfully")
    
    def get_brain_regions_by_pixel(self) -> None:
        """
        Segment brain template into anatomical regions using K-Means clustering.
        
        This method applies K-Means clustering (k=5) to a colored brain template to identify
        and label different anatomical lobes. Manual corrections are applied to improve
        anatomical accuracy based on spatial constraints.
        
        Brain Region Mapping:
            - 0: Background
            - 1: Frontal lobe
            - 2: Parietal lobe  
            - 3: Temporal lobe
            - 4: Occipital lobe
        
        Output:
            - Parcelled_Brain_Template.npy: Region-labeled brain template
            - brain_regions_cmap.png: Color-coded visualization of brain regions
            - BioRender_BrainTemplate_R_Resized.png: Resized original template
        """
        print("Getting brain regions by pixel...")
        
        # Load and resize colored brain template
        img_path = f"{self.brain_template_path}/Brain_Lobes.png"
        brain_img_rsz = self._load_and_resize_image(img_path)
        
        # Save resized template
        plt.figure(figsize=(12, 8))
        plt.imshow(brain_img_rsz)
        plt.axis('off')
        plt.savefig(f"{self.output_path}/BioRender_BrainTemplate_R_Resized.png",
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # Prepare data for clustering
        bgr_img_for_cluster = self._prepare_image_for_clustering(brain_img_rsz)
        
        # Apply K-Means clustering (k=5 for anatomical regions)
        kmeans = KMeans(n_clusters=5, random_state=42).fit(bgr_img_for_cluster)
        
        # Print cluster statistics
        self._print_cluster_statistics(kmeans.labels_, 5)
        
        # Create color mapping for visualization
        cmap_colors = cm.viridis(np.linspace(0, 1, 4))[:, 0:3]
        cmap_colors = (cmap_colors * 255).astype('uint8')
        white_color = [255, 255, 255]
        
        # Background:0, Frontal:1, Parietal:2, Temporal:3, Occipital:4
        clstr_colors = {0: white_color, 1: cmap_colors[0], 2: cmap_colors[1], 
                       3: cmap_colors[2], 4: cmap_colors[3]}
        
        # Initialize output arrays
        clstrd_img = np.zeros_like(brain_img_rsz)
        template_img = np.zeros(brain_img_rsz.shape[0:2])
        
        # Process each row of the image
        for row_idx in range(brain_img_rsz.shape[0]):
            row_pixel_vals = brain_img_rsz[row_idx]
            row_assigned_cltr_labels = kmeans.predict(row_pixel_vals)
            
            for cl in np.unique(row_assigned_cltr_labels):
                clstrd_img[row_idx][row_assigned_cltr_labels == cl] = clstr_colors[cl]
                template_img[row_idx][row_assigned_cltr_labels == cl] = cl
                
                # Apply manual corrections for anatomical accuracy
                if row_idx > 400 and cl == 1:  # Lower frontal -> temporal
                    sel_col = np.zeros(clstrd_img.shape[1], dtype=bool)
                    sel_col[722:1500] = True
                    sel_col[row_assigned_cltr_labels != 1] = False
                    clstrd_img[row_idx][sel_col] = clstr_colors[3]
                    template_img[row_idx][sel_col] = 3
                    
                if row_idx < 400 and cl == 1:  # Upper frontal -> parietal
                    sel_col = np.zeros(clstrd_img.shape[1], dtype=bool)
                    sel_col[1012:1500] = True
                    sel_col[row_assigned_cltr_labels != 1] = False
                    clstrd_img[row_idx][sel_col] = clstr_colors[2]
                    template_img[row_idx][sel_col] = 2
        
        # Save parcelled brain template
        np_parcelled_path = f"{self.output_path}/Parcelled_Brain_Template.npy"
        np.save(np_parcelled_path, template_img)
        
        # Save color-coded visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(clstrd_img)
        plt.axis('off')
        plt.gca().set_facecolor('w')
        plt.savefig(f"{self.output_path}/brain_regions_cmap.png",
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        print("Brain region parcellation completed successfully")
    
    def _calculate_hfo_rates_by_group(self) -> pd.DataFrame:
        """
        Calculate mean HFO occurrence rates grouped by age group and brain region.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['AgeGroup', 'BrainRegion', 'HFO_OccRate']
        """
        if self.df is None:
            raise ValueError("Data not loaded. Cannot calculate HFO rates.")
        
        all_hfo_data = []
        
        for age_group in self.age_groups:
            for brain_region in self.brain_regions:
                hfo_rate = self.df[(self.df['AgeGroup'] == age_group) & 
                                  (self.df['BrainRegion'] == brain_region)]['HFO_OccRate'].mean()
                
                all_hfo_data.append({
                    'AgeGroup': age_group,
                    'BrainRegion': brain_region,
                    'HFO_OccRate': hfo_rate
                })
        
        return pd.DataFrame(all_hfo_data)
    
    def _generate_brain_heatmap(self, age_group: str, hfo_rates: List[float], 
                               norm, output_dir: str, prefix: str, age_index: int) -> None:
        """
        Generate a brain heatmap for a specific age group.
        
        Args:
            age_group (str): Age group name
            hfo_rates (List[float]): HFO rates for each brain region [F, P, T, O]
            norm: Matplotlib normalization object
            output_dir (str): Output directory path
            prefix (str): Filename prefix
            age_index (int): Age group index for filename
        """
        # Load templates
        reference_brain_path = f"{self.output_path}/Clean_Brain_Without_Folds_Template.npy"
        template_path = f"{self.output_path}/Parcelled_Brain_Template.npy"
        
        reference_brain_img = np.load(reference_brain_path)
        reference_brain_img[reference_brain_img == 255] = 230
        
        template_img = np.load(template_path)
        coded_brain_regions_img = np.zeros((template_img.shape[0], template_img.shape[1], 3), dtype='uint8')
        
        # Apply colors based on HFO rates
        colors = self.colormap(norm(hfo_rates))
        colors_int = (colors * 255).astype('uint8')[:, 0:3]
        
        template_coding = {'Background': 0, 'Frontal': 1, 'Parietal': 2, 'Temporal': 3, 'Occipital': 4}
        coded_brain_regions_img[template_img == template_coding['Background']] = [255, 255, 255]
        coded_brain_regions_img[template_img == template_coding['Frontal']] = colors_int[0]
        coded_brain_regions_img[template_img == template_coding['Parietal']] = colors_int[1]
        coded_brain_regions_img[template_img == template_coding['Temporal']] = colors_int[2]
        coded_brain_regions_img[template_img == template_coding['Occipital']] = colors_int[3]
        
        # Create and save figure
        fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
        ax.axis('off')
        ax.set_title(age_group, fontsize=16, fontweight='bold')
        ax.set_facecolor('w')
        
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.colormap), 
                    ax=ax, spacing='proportional', orientation='vertical', 
                    label='HFO Occurrence Rate')
        
        plt.imshow(coded_brain_regions_img, alpha=1)
        plt.imshow(reference_brain_img, cmap='gray', vmin=0, vmax=255, alpha=0.5)
        
        filename = f"{prefix}_{age_index}_{age_group}_brain_regions_cmap_normalized_across_age_groups.png"
        plt.savefig(f"{output_dir}/{filename}", bbox_inches='tight', dpi=150)
        plt.close()
    
    def get_hfo_heatmaps_per_age_group(self) -> None:
        """
        Generate HFO heatmaps normalized within each age group.
        
        Creates brain heatmaps where colors are normalized relative to the HFO rate
        range within each individual age group. This visualization emphasizes
        relative differences between brain regions within each developmental stage.
        
        Output:
            - Individual heatmap images in Norm_Within_Group/ directory
            - Each image shows HFO patterns for one age group with within-group normalization
        """
        print("Getting HFO heatmaps per age group...")
        
        # Calculate HFO rates
        all_hfo_data_df = self._calculate_hfo_rates_by_group()
        
        # Create output directory
        output_dir = f"{self.output_path}/Norm_Within_Group"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate heatmap for each age group
        for age_index, age_group in enumerate(self.age_groups):
            age_group_data = all_hfo_data_df[all_hfo_data_df['AgeGroup'] == age_group]
            hfo_rates = [age_group_data[age_group_data['BrainRegion'] == region]['HFO_OccRate'].iloc[0] 
                        for region in self.brain_regions]
            
            # Normalize within this age group
            norm = mpl.colors.Normalize(vmin=min(hfo_rates), vmax=max(hfo_rates), clip=False)
            
            self._generate_brain_heatmap(age_group, hfo_rates, norm, output_dir, "A", age_index)
        
        print("Within-group HFO heatmaps completed successfully")
    
    def get_hfo_heatmaps_across_age_groups(self) -> None:
        """
        Generate HFO heatmaps normalized across all age groups.
        
        Creates brain heatmaps where colors are normalized relative to the HFO rate
        range across ALL age groups. This visualization allows direct comparison
        of absolute HFO levels between different developmental stages.
        
        Output:
            - Individual heatmap images in Norm_Across_Age_Groups/ directory
            - Bar chart comparing HFO rates across age groups
            - Each image uses the same color scale for cross-age comparison
        """
        print("Getting HFO heatmaps across age groups...")
        
        # Calculate HFO rates
        all_hfo_data_df = self._calculate_hfo_rates_by_group()
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(data=all_hfo_data_df, x='AgeGroup', y='HFO_OccRate', hue='BrainRegion')
        plt.title('HFO Occurrence Rate Across Age Groups', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/HFO_OccRate_Barchart_Across_Age_Groups.png", dpi=150)
        plt.close()
        
        # Create output directory
        output_dir = f"{self.output_path}/Norm_Across_Age_Groups"
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize across all age groups
        norm = mpl.colors.Normalize(vmin=all_hfo_data_df['HFO_OccRate'].min(), 
                                   vmax=all_hfo_data_df['HFO_OccRate'].max(), clip=False)
        
        # Generate heatmap for each age group
        for age_index, age_group in enumerate(self.age_groups):
            age_group_data = all_hfo_data_df[all_hfo_data_df['AgeGroup'] == age_group]
            hfo_rates = [age_group_data[age_group_data['BrainRegion'] == region]['HFO_OccRate'].iloc[0] 
                        for region in self.brain_regions]
            
            self._generate_brain_heatmap(age_group, hfo_rates, norm, output_dir, "B", age_index)
        
        print("Across-group HFO heatmaps completed successfully")
    
    def get_hfo_heatmaps_across_sex_groups(self) -> None:
        """
        Generate HFO heatmaps comparing patterns between sex groups.
        
        Creates brain heatmaps showing HFO patterns separately for males and females,
        allowing comparison of sex-based differences in neural oscillation patterns.
        
        Output:
            - Individual heatmap images in Norm_Across_Sex_Groups/ directory
            - Bar chart comparing HFO rates between sexes
            - Separate brain heatmaps for males (Sex=1) and females (Sex=2)
        """
        print("Getting HFO heatmaps across sex groups...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Cannot generate sex-based heatmaps.")
        
        # Group by sex and brain region
        all_hfo_data_df = self.df.groupby(by=['Sex', 'BrainRegion'], as_index=False).HFO_OccRate.mean()
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=all_hfo_data_df, x='Sex', y='HFO_OccRate', hue='BrainRegion')
        plt.title('HFO Occurrence Rate Across Sex Groups', fontsize=14, fontweight='bold')
        plt.xlabel('Sex (1=Male, 2=Female)')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/HFO_OccRate_Barchart_Across_Sex_Groups.png", dpi=150)
        plt.close()
        
        # Create output directory
        output_dir = f"{self.output_path}/Norm_Across_Sex_Groups"
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize across all sex groups
        norm = mpl.colors.Normalize(vmin=all_hfo_data_df['HFO_OccRate'].min(),
                                   vmax=all_hfo_data_df['HFO_OccRate'].max(), clip=False)
        
        sex_group_names = {1: 'Males', 2: 'Females'}
        sex_groups = all_hfo_data_df['Sex'].unique()
        
        # Generate heatmap for each sex group
        for sex_index, sex_group in enumerate(sex_groups):
            sex_group_data = all_hfo_data_df[all_hfo_data_df['Sex'] == sex_group]
            hfo_rates = [sex_group_data[sex_group_data['BrainRegion'] == region]['HFO_OccRate'].iloc[0] 
                        for region in self.brain_regions]
            
            self._generate_brain_heatmap(sex_group_names[sex_group], hfo_rates, norm, 
                                       output_dir, str(sex_index), sex_index)
        
        print("Sex-based HFO heatmaps completed successfully")
    
    def make_video_within_age_groups(self) -> None:
        """
        Create animated video showing HFO changes across age groups (within-group normalized).
        
        Generates an AVI video file that animates through the within-group normalized
        brain heatmaps, showing how HFO patterns evolve across developmental stages.
        
        Output:
            - Brain_Heatmap_Norm_Within_Group.avi: Animated video file
        """
        print("Making video within age groups...")
        
        image_folder = f"{self.output_path}/Norm_Within_Group/"
        video_name = f"{image_folder}Brain_Heatmap_Norm_Within_Group.avi"
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()
        
        if not images:
            print("No images found for video creation")
            return
        
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        
        cv2.destroyAllWindows()
        video.release()
        
        print(f"Video saved: {video_name}")
    
    def make_video_across_age_groups(self) -> None:
        """
        Create animated video showing HFO changes across age groups (across-group normalized).
        
        Generates an AVI video file that animates through the across-group normalized
        brain heatmaps, allowing comparison of absolute HFO levels across development.
        
        Output:
            - Brain_Heatmap_Norm_Across_Age_Groups.avi: Animated video file
        """
        print("Making video across age groups...")
        
        image_folder = f"{self.output_path}/Norm_Across_Age_Groups/"
        video_name = f"{image_folder}Brain_Heatmap_Norm_Across_Age_Groups.avi"
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()
        
        if not images:
            print("No images found for video creation")
            return
        
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        
        cv2.destroyAllWindows()
        video.release()
        
        print(f"Video saved: {video_name}")
    
    def run_complete_analysis(self) -> None:
        """
        Execute the complete brain heatmap analysis pipeline.
        
        This method runs all analysis steps in the correct order:
        1. Clean brain template (remove folds)
        2. Parcellate brain regions
        3. Generate within-group HFO heatmaps
        4. Generate across-group HFO heatmaps
        5. Generate sex-based HFO heatmaps
        6. Create animated videos
        
        This is the main entry point for running the full analysis.
        """
        print("Starting complete brain heatmap analysis pipeline...")
        
        try:
            # Core analysis steps
            self.clean_brain_with_folds()
            self.get_brain_regions_by_pixel()
            self.get_hfo_heatmaps_per_age_group()
            self.get_hfo_heatmaps_across_age_groups()
            self.get_hfo_heatmaps_across_sex_groups()
            
            # Video generation
            self.make_video_within_age_groups()
            self.make_video_across_age_groups()
            
            print("\\nComplete analysis pipeline finished successfully!")
            print(f"Results saved in: {self.output_path}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise


def main():
    """
    Main function to run the complete brain heatmap analysis.
    
    Creates a BrainHeatmapAnalyzer instance and runs the full analysis pipeline.
    """
    # Initialize analyzer with default parameters
    analyzer = BrainHeatmapAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
