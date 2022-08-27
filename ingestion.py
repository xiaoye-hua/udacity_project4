import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']




#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    file_list = [os.path.join(input_folder_path, name) for name in os.listdir(input_folder_path) if '.csv' in name]
    df_lst = [pd.read_csv(name) for name in file_list]
    df = pd.concat(df_lst)
    df = df.drop_duplicates()
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.writelines(','.join(file_list))
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
