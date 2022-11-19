## --- DATA LOADER --- ##

from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import os
#create a class to load daata in batches
class customdataset(Dataset):
    def __init__(self, dataset, img_size, set_type ,transforms):
        self.img_size = img_size
        #read the csv file 
        self.data_info = dataset
        self.set_type = set_type
        #length
        self.data_len = len(self.data_info.index)
        #image_rows
        self.image_rows = dataset.index
        if set_type == "train":
            #image paths 
            self.image_array = np.asarray(self.data_info.iloc[:,4])
            #labels 
            self.label_array = np.asarray(self.data_info.iloc[:,0])
        elif set_type == "test":
            #image paths 
            self.image_array = np.asarray(self.data_info.iloc[:,3])
            #labels 
            self.label_array = np.zeros(len(self.data_info.index))
        
        self.transforms = transforms

    def __getitem__(self,index):
        #image name
        image_name = self.image_array[index]
        image_row =  self.image_rows[index]
        img_as_img = cv2.imread((image_name),0)
        
        img_resized = cv2.resize(img_as_img, (self.img_size, self.img_size))

        #convert 3D tensor to 4D tensor with shape (1, 227, 227, 3) and return 4D tensor
        final_img = np.expand_dims(img_resized, axis=0)
        image_label = self.label_array[index]
        return(final_img, image_label, image_row)

    def __len__(self):
        return self.data_len