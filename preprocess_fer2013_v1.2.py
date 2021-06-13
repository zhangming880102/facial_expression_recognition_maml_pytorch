# create data and label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import h5py

file = 'data/fer2013.csv'

# Creat the list to store the data and label information
Training_x = []
Training_y = []

datapath = os.path.join('data','fer_maml_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

with open(file,'r') as csvin:
    data=csv.reader(csvin)
    for row in data:
        if row[0]=='emotion':
            continue
        temp_list = []
        for pixel in row[1].split():
            temp_list.append(int(pixel))
        I = np.asarray(temp_list)
        Training_y.append(int(row[0]))
        Training_x.append(I.tolist())


print(np.shape(Training_x))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("data_label", dtype = 'int64', data=Training_y)
datafile.close()

print("Save data finish!!!")
