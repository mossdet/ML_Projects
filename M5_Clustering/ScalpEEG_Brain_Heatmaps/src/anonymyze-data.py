import pandas as pd
from pathlib import Path


hfo_data_fpath = Path('M5_Clustering/ScalpEEG_Brain_Heatmaps/Data/HFO_characterized_channels_STRATIFIED_RF_dlp.csv')

if not hfo_data_fpath.exists():
    raise FileNotFoundError(f"File not found: {hfo_data_fpath}")

# Read the CSV file
df = pd.read_csv(hfo_data_fpath)

for pat_nr, pat_name in enumerate(df['PatientName'].unique()):
    df.loc[:, 'PatientName'] = f'Patient_{pat_nr + 1}'
    df.loc[:, 'his_id'] = pat_nr+1
    df.loc[:, 'FileName'] = f'Patient_{pat_nr + 1}'

# Save the anonymized DataFrame to a new CSV file
anonymized_fpath = hfo_data_fpath.parent / 'anonymized_data.csv'
df.to_csv(anonymized_fpath, index=False)
pass
