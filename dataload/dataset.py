import torch
import numpy as np
import nibabel as nib
import os 
from itertools import product
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset
import os 

class dataset_train(Dataset):

    def __init__(self,image_dir,label_dir,crop_size = 64,num_samples = 10,transform = None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.ID_list = sorted([image_data.split("-")[0] for image_data in os.listdir(image_dir)])
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.transform = transform

    def positive_centroids(self,label_arr):
        pos_centroids = [tuple([round(index) for index in every.centroid]) for every in regionprops(label_arr)]
        return pos_centroids

    def spine_negtive_centroids(self,shape, crop_size, num_samples):
        x_start = shape[0] // 2 - 40
        x_end = shape[0] // 2 + 40
        y_start = 300
        y_end = 400
        z_start = crop_size // 2
        z_end = shape[2] - crop_size // 2
        spine_neg_centroids = [(np.random.randint(x_start, x_end),np.random.randint(y_start, y_end),np.random.randint(z_start, z_end))
        for i in range(num_samples)]
        return spine_neg_centroids

    def symmetric_negtive_centroids(self,positive_centroids, shape_x):
        symmetric_neg_centroids = [(shape_x - x, y, z) for x, y, z in positive_centroids]
        return symmetric_neg_centroids

    def negtive_centroids(self, positive_centroids, image_shape):
        num_positive = len(positive_centroids)
        symmetric_neg_centroids = self.symmetric_negtive_centroids(positive_centroids, image_shape[0])
        if num_positive < self.num_samples // 2:
            spine_neg_centroids = self.spine_negtive_centroids(image_shape,self.crop_size, self.num_samples - 2 * num_positive)
        else:
            spine_neg_centroids = self.spine_negtive_centroids(image_shape,self.crop_size, num_positive)
        return symmetric_neg_centroids + spine_neg_centroids

    def all_centroids(self, label_arr):
        pos_centroids = self.positive_centroids(label_arr)
        neg_centroids = self.negtive_centroids(pos_centroids,label_arr.shape)
        num_positive = len(pos_centroids)
        num_negtive = len(neg_centroids)
        if num_positive >= self.num_samples:
            num_positive = self.num_samples // 2
            num_negtive = self.num_samples // 2
        elif num_positive >= self.num_samples // 2:
            num_negtive = self.num_samples - num_positive
        if num_positive < len(pos_centroids):
            pos_centroids = [pos_centroids[i] for i in np.random.choice(range(0, len(pos_centroids)), size=num_positive, replace=False)]
        if num_negtive < len(neg_centroids):
            neg_centroids = [neg_centroids[i] for i in np.random.choice(range(0, len(neg_centroids)), size=num_negtive, replace=False)]
        all_centroids = pos_centroids + neg_centroids

        return all_centroids

    def cut_crop(self, image, centroid):
        all = np.ones(tuple([self.crop_size] * 3)) * (-1024)
        image_start = [max(0, int(centroid[i] - self.crop_size // 2)) for i in range(len(centroid))]
        image_end = [min(image.shape[i], int(centroid[i] + self.crop_size // 2)) for i in range(len(centroid))]
        result_start = [max(0, self.crop_size // 2 - centroid[i]) for i in range(len(centroid))]
        result_end = [min(image.shape[i] - (centroid[i] - self.crop_size // 2),self.crop_size) for i in range(len(centroid))]
        
        all[
            result_start[0]:result_end[0],result_start[1]:result_end[1],result_start[2]:result_end[2],
        ] = image[
        
            image_start[0]:image_end[0],image_start[1]:image_end[1],image_start[2]:image_end[2],
        ]
        return all

    def TransForms(self, image):
        for trans in self.transform:
            image = trans(image)
        return image
  
    def __getitem__(self, index):
        ID = self.ID_list[index]
        image_path = os.path.join(self.image_dir, f"{ID}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{ID}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_arr = image.get_fdata().astype(np.float)
        label_arr = label.get_fdata().astype(np.uint8)
        all_centroids = self.all_centroids(label_arr)
        image_crops = [self.cut_crop(image_arr, centroid) for centroid in all_centroids]
        label_crops = [self.cut_crop(label_arr, centroid) for centroid in all_centroids]
        if self.transform is not None:
            image_crops = [self.TransForms(image_crop) for image_crop in image_crops]
        image_crops = torch.tensor(np.stack(image_crops)[:, np.newaxis],dtype=torch.float)
        label_crops = (np.stack(label_crops) > 0).astype(np.float)
        label_crops = torch.tensor(label_crops[:, np.newaxis],dtype=torch.float)
        return image_crops, label_crops

    @staticmethod
    def _collate_fn(batch_samples):
        image_crops = torch.cat([sample[0] for sample in batch_samples])
        label_crops = torch.cat([sample[1] for sample in batch_samples])
        return image_crops, label_crops

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,num_workers=num_workers, collate_fn=dataset_train._collate_fn)

    def __len__(self):
        return len(self.ID_list)


class dataset_test(Dataset):
    def __init__(self,image_path,crop_size = 64,transforms = None):
        image = nib.load(image_path)
        self.image = image.get_fdata().astype(np.int16)
        self.image_affine = image.affine
        self.crop_size = crop_size
        self.transforms = transforms
        self.centers = self.calculate_centers()

    def calculate_centers(self):
        center_dims = [list(range(0, dim, self.crop_size//2))[1:-1] + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*center_dims))
        return centers

    def __len__(self):
        return len(self.centers)

    def devide_crops(self, index):
        center_x, center_y, center_z = self.centers[index]
        crops = self.image[center_x - self.crop_size // 2:center_x + self.crop_size // 2,center_y - self.crop_size // 2:center_y + self.crop_size // 2,center_z - self.crop_size // 2:center_z + self.crop_size // 2]
        return crops

    def TransForms(self, image):
        for trans in self.transforms:
            image = trans(image)
        return image

    def __getitem__(self, index):
        image = self.devide_crops(index)
        center = self.centers[index]
        if self.transforms is not None:
            image = self.TransForms(image)
        image = torch.tensor(image[np.newaxis], dtype=torch.float)
        return image, center

    @staticmethod
    def _collate_fn(batch_samples):
        images = torch.stack([sample[0] for sample in batch_samples])
        centers = [sample[1] for sample in batch_samples]
        return images, centers

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, num_workers=num_workers,collate_fn=dataset_test._collate_fn)

