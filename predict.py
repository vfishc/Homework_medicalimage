import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm
from dataload.dataset import dataset_test
from dataload import transforms as tsfm
from model.model import fracnet




def delete_low(pre, prob_thresh):
    pre = np.where(pre > prob_thresh, pre, 0)
    return pre


def delete_spine(pre, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pre)


def delete_small(pre, size_thresh):
    pre_bin = pre > 0
    pre_bin = remove_small_objects(pre_bin, size_thresh)
    pre = np.where(pre_bin, pre, 0)
    return pre


def post_process(pre, image, prob_thresh, bone_thresh, size_thresh):

    pre = delete_low(pre, prob_thresh)
    pre = delete_spine(pre, image, bone_thresh)
    pre = delete_small(pre, size_thresh)
    return pre


def single_process(model, dataloader, if_postprocess, prob_thresh, bone_thresh, size_thresh):
    pre = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)
            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur = pre[center_x - crop_size // 2:center_x + crop_size // 2,center_y - crop_size // 2:center_y + crop_size // 2,center_z - crop_size // 2:center_z + crop_size // 2]
                pre[center_x - crop_size // 2:center_x + crop_size // 2,center_y - crop_size // 2:center_y + crop_size // 2,center_z - crop_size // 2:center_z + crop_size // 2] =np.where(cur > 0, np.mean((output[i],cur), axis=0), output[i])

    if if_postprocess:
        pre = post_process(pre, dataloader.dataset.image, prob_thresh, bone_thresh, size_thresh)
  
    return pre


def submission(pre, image_id, affine):
    pre_label = label(pre > 0).astype(np.int16)  
    lung_mask = nib.load('/home/yliu/weiyuxi/data_test/ribfrac-test-images/segment/'+image_id[7:10]+'-lung.nii.gz')
    lung = lung_mask.get_fdata().astype(np.int8)
   
    pre_regions = regionprops(pre_label, pre)
    bone_size = 20
    for region in pre_regions:
        x = int(region.centroid[0])
        y = int(region.centroid[1])
        z = int(region.centroid[2])
        if (lung[min(x+bone_size,pre.shape[0]-1)][min(y+bone_size,pre.shape[1]-1)][min(z+bone_size,pre.shape[2]-1)] == 0 \
            and lung[min(x+bone_size,pre.shape[0]-1)][min(y+bone_size,pre.shape[1]-1)][max(z-bone_size,0)] == 0\
            and lung[min(x+bone_size,pre.shape[0]-1)][max(y-bone_size,0)][min(z+bone_size,pre.shape[2]-1)] == 0\
            and lung[max(x-bone_size,0)][min(y+bone_size,pre.shape[1]-1)][min(z+bone_size,pre.shape[2]-1)] == 0\
            and lung[max(x-bone_size,0)][max(y-bone_size,0)][min(z+bone_size,pre.shape[2]-1)] == 0\
            and lung[max(x-bone_size,0)][min(y+bone_size,pre.shape[1]-1)][max(z-bone_size,0)] == 0\
            and lung[min(x+bone_size,pre.shape[0]-1)][max(y-bone_size,0)][max(z-bone_size,0)] == 0\
            and lung[max(x-bone_size,0)][max(y-bone_size,0)][max(z-bone_size,0)] == 0):
          
            for coord in region.coords:
                pre[tuple(coord)] = 0
    pre_index = [0] + [region.label for region in pre_regions]
    pre_proba = [0.0] + [region.mean_intensity for region in pre_regions]
    pre_label_code = [0] + [1] * int(pre_label.max())
    pre_image = nib.Nifti1Image(pre_label, affine)
    pre_info = pd.DataFrame({"public_id": [image_id] * len(pre_index),"label_id": pre_index,"confidence": pre_proba,"label_code": pre_label_code})
    return pre_image, pre_info


def predict(args):
    batch_size = 16
    num_workers = 4
    postprocess = True if args.postprocess == "True" else False

    #torch.cuda.set_device(1)
    
    model = fracnet()
    model.eval()
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model=nn.DataParallel(model.cuda())

    transforms = [tsfm.win(-200, 1000),tsfm.norm(-200, 1000)]

    image_path_list = sorted([os.path.join(args.image_dir, file) for file in os.listdir(args.image_dir) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0] for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pre_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = dataset_test(image_path, transforms=transforms)
        dataloader = dataset_test.get_dataloader(dataset, batch_size, num_workers)
        pre_arr =single_process(model, dataloader, postprocess, args.prob_thresh, args.bone_thresh, args.size_thresh)
        pre_image, pre_info = submission(pre_arr, image_id, dataset.image_affine)
        pre_info_list.append(pre_info)
        pre_path = os.path.join(args.pre_dir, f"{image_id}_pre.nii.gz")
        nib.save(pre_image, pre_path)

        progress.update()

    pre_info = pd.concat(pre_info_list, ignore_index=True)
    pre_info.to_csv(os.path.join(args.pre_dir, "pre_info.csv"),index=False)


if __name__ == "__main__":
    import argparse


    prob_thresh = 0.15
    bone_thresh = 400
    size_thresh = 800

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True,
        help="The image nii directory.")
    parser.add_argument("--pre_dir", required=True,
        help="The directory for saving predictions.")
    parser.add_argument("--model_path", default=None,
        help="The PyTorch model weight path.")
    parser.add_argument("--prob_thresh", default=0.1,
        help="Prediction probability threshold.")
    parser.add_argument("--bone_thresh", default=400,
        help="Bone binarization threshold.")
    parser.add_argument("--size_thresh", default=800,
        help="Prediction size threshold.")
    parser.add_argument("--postprocess", default="True",
        help="Whether to execute post-processing.")
    args = parser.parse_args()
    predict(args)
