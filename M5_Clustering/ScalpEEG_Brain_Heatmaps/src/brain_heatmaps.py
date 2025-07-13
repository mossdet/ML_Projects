import matplotlib as mpl
mpl.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
import pandas as pd
import os
import seaborn as sns

from sklearn.cluster import KMeans

from matplotlib import cm

DATA_PATH = "/home/dlp/Development/ML_Projects/M5_Clustering/ScalpEEG_Brain_Heatmaps/Data"
BRAIN_TEMPLATE_PATH = "/home/dlp/Development/ML_Projects/M5_Clustering/ScalpEEG_Brain_Heatmaps/Brain_Template"
OUTPUT_PATH = "/home/dlp/Development/ML_Projects/M5_Clustering/ScalpEEG_Brain_Heatmaps/Output"

COLORMAP = mpl.cm.viridis
#COLORMAP = mpl.cm.Reds

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(BRAIN_TEMPLATE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

img_resize_width = 2000
img_resize_height = 1300

def clean_brain_with_folds():

    print("Cleaning brain with folds...")

    img_path = f"{BRAIN_TEMPLATE_PATH}/Brain_With_Folds.png"
    brain_img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    brain_img_rsz = cv2.resize(brain_img, (img_resize_width, img_resize_height), interpolation=cv2.INTER_LINEAR)

    # Reshape the image to be a list of pixels
    r,g,b = cv2.split(brain_img_rsz)
    bgr_img_for_cluster = np.transpose(np.vstack((r.ravel(), g.ravel(), b.ravel())))
    kmeans = KMeans(n_clusters=4, random_state=42).fit(bgr_img_for_cluster)
    cluster_labels = kmeans.labels_
    np.unique(cluster_labels)

    print("Cluster 0 percentage:", np.sum(kmeans.labels_==0)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 1 percentage:", np.sum(kmeans.labels_==1)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 2 percentage:", np.sum(kmeans.labels_==2)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 3 percentage:", np.sum(kmeans.labels_==3)/bgr_img_for_cluster.shape[0]*100)
    pass

    clean_image = np.zeros_like(cluster_labels)
    clean_image[cluster_labels==0] = 255
    clean_image[cluster_labels==1] = 220
    clean_image[cluster_labels==2] = 128
    clean_image[cluster_labels==3] = 220
    clean_image = clean_image.reshape(brain_img_rsz.shape[0], brain_img_rsz.shape[1])

    np_brain_with_folds_template_fn = f"{OUTPUT_PATH}/Clean_Brain_Without_Folds_Template.npy"
    np.save(np_brain_with_folds_template_fn, clean_image)


    plt.imshow(clean_image, cmap='gray', vmin=0, vmax=255)
    ax = plt.gca()
    ax.set_facecolor('w')
    plt.axis('off')
    plt.savefig(f"{OUTPUT_PATH}/Clean_Brain_Without_Folds_Template.png")
    #plt.show()
    plt.close()
    pass


def get_brain_regions_by_pixel():

    print("Getting brain regions by pixel...")

    # Image with colored lobes
    img_path = f"{BRAIN_TEMPLATE_PATH}/Brain_Lobes.png"
    brain_img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    brain_img_rsz = cv2.resize(brain_img, (img_resize_width, img_resize_height), interpolation=cv2.INTER_LINEAR)
    plt.imshow(brain_img_rsz)
    plt.savefig(f"{OUTPUT_PATH}/BioRender_BrainTemplate_R_Resized.png")
    #plt.close()


    # Reshape the image to be a list of pixels
    r,g,b = cv2.split(brain_img_rsz)
    bgr_img_for_cluster = np.transpose(np.vstack((r.ravel(), g.ravel(), b.ravel())))
    kmeans = KMeans(n_clusters=5, random_state=42).fit(bgr_img_for_cluster)
    cluster_labels = kmeans.labels_
    np.unique(cluster_labels)

    print("Cluster 0 percentage:", np.sum(kmeans.labels_==0)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 1 percentage:", np.sum(kmeans.labels_==1)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 2 percentage:", np.sum(kmeans.labels_==2)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 3 percentage:", np.sum(kmeans.labels_==3)/bgr_img_for_cluster.shape[0]*100)
    print("Cluster 4 percentage:", np.sum(kmeans.labels_==4)/bgr_img_for_cluster.shape[0]*100)

    pass


    cmap_colors = cm.viridis(np.linspace(0, 1, 4))[:,0:3]
    cmap_colors = (cmap_colors*255).astype('uint8')
    white_color = [255, 255, 255]

    # Background:0, Frontal:1, Parietal:2, Temporal:3, Occipital:4
    clstr_colors = {0:white_color, 1:cmap_colors[0], 2:cmap_colors[1], 3:cmap_colors[2], 4:cmap_colors[3]}
    clstrd_img = np.zeros_like(brain_img_rsz)
    template_img = np.zeros(brain_img_rsz.shape[0:2])
    for row_idx in range(brain_img_rsz.shape[0]):
        row_pixel_vals = brain_img_rsz[row_idx]
        row_assigned_cltr_labels = kmeans.predict(row_pixel_vals)
        for cl in np.unique(row_assigned_cltr_labels):
            clstrd_img[row_idx][row_assigned_cltr_labels==cl] = clstr_colors[cl]
            
            template_img[row_idx][row_assigned_cltr_labels==cl] = cl
            if row_idx > 400 and cl==1:
                #clstrd_img[row_idx][722:1500] = clstr_colors[3]
                sel_col = np.zeros(clstrd_img.shape[1], dtype=bool)
                sel_col[722:1500] = True
                sel_col[row_assigned_cltr_labels!=1] = False
                clstrd_img[row_idx][sel_col] = clstr_colors[3]
                template_img[row_idx][sel_col] = 3
                
            if row_idx < 400 and cl==1:
                sel_col = np.zeros(clstrd_img.shape[1], dtype=bool)
                sel_col[1012:1500] = True
                sel_col[row_assigned_cltr_labels!=1] = False
                clstrd_img[row_idx][sel_col] = clstr_colors[2]
                template_img[row_idx][sel_col] = 2

    flat_template_img = template_img.flatten()
    template_img = flat_template_img.reshape(template_img.shape)

    np_parcelled_brain_template_fn = f"{OUTPUT_PATH}/Parcelled_Brain_Template.npy"
    np.save(np_parcelled_brain_template_fn, template_img)

    ax = plt.gca()
    ax.set_facecolor('w')

    plt.axis('off')
    plt.imshow(clstrd_img)
    plt.savefig(f"{OUTPUT_PATH}/brain_regions_cmap.png")
    #plt.show()
    plt.close()
    pass


def get_hfo_heatmaps_per_age_group():

    print("Getting HFO heatmaps per age group...")

    # Image without colored lobes
    np_brain_with_folds_template_fn = f"{OUTPUT_PATH}/Clean_Brain_Without_Folds_Template.npy"
    reference_brain_img = np.load(np_brain_with_folds_template_fn)
    reference_brain_img[reference_brain_img==255] = 230

    df = pd.read_csv(f"{DATA_PATH}/anonymized_data.csv")
    age_groups = ['1monto2yrs', '3to5yrs', '6to10yrs', '11to13yrs', '14to17yrs']
    brain_regions = ['F', 'P', 'T', 'O']

    template_img = np.load(f"{OUTPUT_PATH}/Parcelled_Brain_Template.npy")
    coded_brain_regions_img = np.zeros((template_img.shape[0], template_img.shape[1], 3), dtype='uint8')

    all_hfo_data_df = pd.DataFrame()
    for agi, age_group in enumerate(age_groups):
        age_group_hfo_data = {'AgeGroup':[], 'BrainRegion':[], 'HFO_OccRate':[]}
        
        for bri, brain_region in enumerate(brain_regions):
            hfo_rate = df[(df['AgeGroup']==age_group) & (df['BrainRegion']==brain_region)]['HFO_OccRate'].mean()
            age_group_hfo_data['AgeGroup'].append(age_group)
            age_group_hfo_data['BrainRegion'].append(brain_region)
            age_group_hfo_data['HFO_OccRate'].append(hfo_rate)

        all_hfo_data_df = pd.concat([all_hfo_data_df, pd.DataFrame(age_group_hfo_data)], axis=0)

    cmap = COLORMAP


    for agi, age_group in enumerate(age_groups):
        
        age_group_hfo_rates = all_hfo_data_df.HFO_OccRate[all_hfo_data_df['AgeGroup']==age_group]   
        norm = mpl.colors.Normalize(vmin=np.min(age_group_hfo_rates), vmax=np.max(age_group_hfo_rates), clip=False)
        #norm = mpl.colors.Normalize(vmin=0, vmax=np.max(age_group_hfo_rates.HFO_OccRate), clip=False)

        fig, ax = plt.subplots(layout='constrained')
        colors = cmap(norm(age_group_hfo_rates))
        colors_int = (colors*255).astype('uint8')[:,0:3]

        template_coding = {'Background':0, 'Frontal':1, 'Parietal':2, 'Temporal':3, 'Occipital':4}
        coded_brain_regions_img[template_img == template_coding['Background']] = [255, 255, 255]
        coded_brain_regions_img[template_img == template_coding['Frontal']] = colors_int[0]
        coded_brain_regions_img[template_img == template_coding['Parietal']] = colors_int[1]
        coded_brain_regions_img[template_img == template_coding['Temporal']] = colors_int[2]
        coded_brain_regions_img[template_img == template_coding['Occipital']] = colors_int[3]

        ax.axis('off')
        ax.set_title(age_group)
        ax.set_facecolor('w')

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), spacing='proportional', orientation='vertical', label='HFO Occ.Rate')
        plt.imshow(coded_brain_regions_img, alpha=1)
        plt.imshow(reference_brain_img, cmap='gray', vmin=0, vmax=255, alpha=0.5)

        fig_fpath = f"{OUTPUT_PATH}/Norm_Within_Group/"
        os.makedirs(fig_fpath, exist_ok=True)

        plt.savefig(f"{fig_fpath}/A_{agi}_{age_group}_brain_regions_cmap_normalized_within_age_group.png")
        #plt.show()
        plt.close()
        pass


def get_hfo_heatmaps_across_age_groups():

    print("Getting HFO heatmaps across age groups...")

    # Image without colored lobes
    np_brain_with_folds_template_fn = f"{OUTPUT_PATH}/Clean_Brain_Without_Folds_Template.npy"
    reference_brain_img = np.load(np_brain_with_folds_template_fn)
    reference_brain_img[reference_brain_img==255] = 255

    df = pd.read_csv(f"{DATA_PATH}/anonymized_data.csv")
    age_groups = ['1monto2yrs', '3to5yrs', '6to10yrs', '11to13yrs', '14to17yrs']
    brain_regions = ['F', 'P', 'T', 'O']

    template_img = np.load(f"{OUTPUT_PATH}/Parcelled_Brain_Template.npy")
    coded_brain_regions_img = np.zeros((template_img.shape[0], template_img.shape[1], 3), dtype='uint8')

    all_hfo_data_df = pd.DataFrame()
    for agi, age_group in enumerate(age_groups):
        age_group_hfo_data = {'AgeGroup':[], 'BrainRegion':[], 'HFO_OccRate':[]}
        
        for bri, brain_region in enumerate(brain_regions):
            hfo_rate = df[(df['AgeGroup']==age_group) & (df['BrainRegion']==brain_region)]['HFO_OccRate'].mean()
            age_group_hfo_data['AgeGroup'].append(age_group)
            age_group_hfo_data['BrainRegion'].append(brain_region)
            age_group_hfo_data['HFO_OccRate'].append(hfo_rate)

        all_hfo_data_df = pd.concat([all_hfo_data_df, pd.DataFrame(age_group_hfo_data)], axis=0)


    sns.barplot(data=all_hfo_data_df, x='AgeGroup', y='HFO_OccRate', hue='BrainRegion')
    plt.savefig(f"{OUTPUT_PATH}/HFO_OccRate_Barchart_Across_Age_Groups.png")

    cmap = COLORMAP

    #norm = mpl.colors.Normalize(vmin=np.min(all_hfo_data_df.HFO_OccRate), vmax=np.max(all_hfo_data_df.HFO_OccRate), clip=False)
    #all_hfo_data_df.HFO_OccRate = all_hfo_data_df.HFO_OccRate
    norm = mpl.colors.Normalize(vmin=np.min(all_hfo_data_df.HFO_OccRate), vmax=np.max(all_hfo_data_df.HFO_OccRate), clip=False)

    for agi, age_group in enumerate(age_groups):
        age_group_hfo_rates = all_hfo_data_df.HFO_OccRate[all_hfo_data_df['AgeGroup']==age_group]   
        fig, ax = plt.subplots(layout='constrained')
        colors = cmap(norm(age_group_hfo_rates))
        colors_int = (colors*255).astype('uint8')[:,0:3]
        #colors_int = np.array([np.append(c, 255) for c in colors_int])

        template_coding = {'Background':0, 'Frontal':1, 'Parietal':2, 'Temporal':3, 'Occipital':4}
        coded_brain_regions_img[template_img == template_coding['Background']] = [255, 255, 255]
        coded_brain_regions_img[template_img == template_coding['Frontal']] = colors_int[0]
        coded_brain_regions_img[template_img == template_coding['Parietal']] = colors_int[1]
        coded_brain_regions_img[template_img == template_coding['Temporal']] = colors_int[2]
        coded_brain_regions_img[template_img == template_coding['Occipital']] = colors_int[3]

        ax.axis('off')
        ax.set_title(age_group)
        ax.set_facecolor('w')
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), spacing='proportional', orientation='vertical', label='HFO Occ.Rate')
        plt.imshow(coded_brain_regions_img, alpha=1)
        plt.imshow(reference_brain_img, cmap='gray', vmin=0, vmax=255, alpha=0.5)

        fig_fpath = f"{OUTPUT_PATH}/Norm_Across_Age_Groups/"
        os.makedirs(fig_fpath, exist_ok=True)

        plt.savefig(f"{fig_fpath}/B_{agi}_{age_group}_brain_regions_cmap_normalized_across_age_groups.png")
        #plt.show()
        plt.close()
        pass

def get_hfo_heatmaps_across_sex_groups():

    print("Getting HFO heatmaps across Sex groups...")

    # Image without colored lobes but with sulci
    np_brain_with_folds_template_fn = f"{OUTPUT_PATH}/Clean_Brain_Without_Folds_Template.npy"
    reference_brain_img = np.load(np_brain_with_folds_template_fn)
    reference_brain_img[reference_brain_img==255] = 255

    # Image with colored lobes but no sulci
    template_img = np.load(f"{OUTPUT_PATH}/Parcelled_Brain_Template.npy")
    coded_brain_regions_img = np.zeros((template_img.shape[0], template_img.shape[1], 3), dtype='uint8')

    # HFO Occ.Rate data
    df = pd.read_csv(f"{DATA_PATH}/anonymized_data.csv")

    # Group by Sex and Brain Region
    all_hfo_data_df = df.groupby(by=['Sex', 'BrainRegion'], as_index=False).HFO_OccRate.mean()

    # Plot bar chart with same information as brain heatmap
    sns.barplot(data=all_hfo_data_df, x='Sex', y='HFO_OccRate', hue='BrainRegion')
    plt.savefig(f"{OUTPUT_PATH}/HFO_OccRate_Barchart_Across_Sex_Groups.png")

    cmap = COLORMAP
    norm = mpl.colors.Normalize(vmin=np.min(all_hfo_data_df.HFO_OccRate), vmax=np.max(all_hfo_data_df.HFO_OccRate), clip=False)
    brain_regions_ordered = ['F', 'P', 'T', 'O'] # Musr be the same as teh order given to the colors below
    sex_groups = all_hfo_data_df.Sex.unique()

    sex_group_names_dict = {1:'Males', 2:'Females'}

    for sgi, sex_group in enumerate(sex_groups):
        sex_group_hfo_rates = []
        for bri, brain_region in enumerate(brain_regions_ordered):
            hfo_rates = all_hfo_data_df[(all_hfo_data_df['Sex']==sex_group) & (all_hfo_data_df['BrainRegion']==brain_region)].HFO_OccRate.values
            assert len(hfo_rates) == 1, "There should be only one value for the HFO rate"
            sex_group_hfo_rates.append(hfo_rates[0])
        colors = cmap(norm(sex_group_hfo_rates))
        colors_int = (colors*255).astype('uint8')[:,0:3]
        #colors_int = np.array([np.append(c, 255) for c in colors_int])

        template_coding = {'Background':0, 'Frontal':1, 'Parietal':2, 'Temporal':3, 'Occipital':4}
        coded_brain_regions_img[template_img == template_coding['Background']] = [255, 255, 255]
        coded_brain_regions_img[template_img == template_coding['Frontal']] = colors_int[0]
        coded_brain_regions_img[template_img == template_coding['Parietal']] = colors_int[1]
        coded_brain_regions_img[template_img == template_coding['Temporal']] = colors_int[2]
        coded_brain_regions_img[template_img == template_coding['Occipital']] = colors_int[3]

        fig, ax = plt.subplots(layout='constrained')
        ax.axis('off')
        ax.set_title(sex_group_names_dict[sex_group])
        ax.set_facecolor('w')
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), spacing='proportional', orientation='vertical', label='HFO Occ.Rate')
        plt.imshow(coded_brain_regions_img, alpha=1)
        plt.imshow(reference_brain_img, cmap='gray', vmin=0, vmax=255, alpha=0.5)

        fig_fpath = f"{OUTPUT_PATH}/Norm_Across_Sex_Groups/"
        os.makedirs(fig_fpath, exist_ok=True)

        plt.savefig(f"{fig_fpath}/{sgi}_{sex_group_names_dict[sex_group]}_brain_regions_cmap_normalized_across_sex_groups.png")
        #plt.show()
        plt.close()
        pass


def make_video_within_age_groups():

    print("Making video within age groups...")

    pass
    image_folder = f"{OUTPUT_PATH}/Norm_Within_Group/"
    video_name = f"{image_folder}Brain_Heatmap_Norm_Within_Group.avi"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    #images.append(images[-1])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def make_video_across_age_groups():

    print("Making video across age groups...")

    pass
    image_folder = f"{OUTPUT_PATH}/Norm_Across_Age_Groups/"
    video_name = f"{image_folder}Brain_Heatmap_Norm_Across_Age_Groups.avi"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    #images.append(images[-1])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    clean_brain_with_folds()
    get_brain_regions_by_pixel()
    get_hfo_heatmaps_per_age_group()
    get_hfo_heatmaps_across_age_groups()
    make_video_within_age_groups()
    make_video_across_age_groups()
    get_hfo_heatmaps_across_sex_groups()