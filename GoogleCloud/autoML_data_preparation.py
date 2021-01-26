import os
import pandas as pd

data_folders = ['daisy',
 'dandelion',
 'roses',
 'sunflowers',
 'tulips']

local_base_path = '/home/enrique/Training_Images/flower_photos/'
# array of arrays, containing the list files, grouped by folder
filenames = [os.listdir(local_base_path+f) for f in data_folders]
[print(f[1]) for f in filenames]
[len(f) for f in filenames]

files_dict = dict(zip(data_folders, filenames))
base_gcs_path = 'gs://flower-classifier-us-central1/'

# gs://cloudml-demo-vcm/chairs_table_bike/chair_black/chair_black157.jpg, 'chair_black'
# base_gcs_path + dict_key + '/' + filename
data_array = []

for (dict_key, files_list) in files_dict.items():
    for filename in files_list:
        #         print(base_gcs_path + dict_key + '/' + filename)
        if '.jpg' not in filename:
            continue  # don't include non-photos

        label = dict_key
        #         label = 'chair' if 'chair' in dict_key else dict_key # for grouping all chairs as one label

        data_array.append((base_gcs_path + dict_key + '/' + filename, label))

print(data_array)
dataframe = pd.DataFrame(data_array)
print(dataframe)
dataframe.to_csv(local_base_path+'all_data.csv', index=False, header=False)