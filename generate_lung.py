from tqdm import tqdm
import numpy as np
import nibabel as nib
from utils.util import *
from segment_lung import segment_lung
from segment_airway import segment_airway

if __name__ == "__main__":
    params = define_parameter()
    for i in tqdm(range(652,653)):
        I = nib.load('/home/yliu/weiyuxi/data_test/ribfrac-test-images/ribfrac-test-images/'+'RibFrac'+str(i)+'-image.nii.gz')
        I_affine = I.affine
        I = I.get_fdata()


        Mlung = segment_lung(params, I, I_affine)


        Mlung, Maw = segment_airway(params, I, I_affine, Mlung,i)
