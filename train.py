import torch
import torch.nn as nn
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from torch import optim
from dataload.dataset import dataset_train
import dataload.transforms as tsfm
from fastai.callbacks.tracker import SaveModelCallback
from utils.tools import dice, recall, precision, fbeta_score
from model.model import fracnet
from model.losses import MixLoss, DiceLoss,FocalLoss,GHMLoss
from functools import partial

import os



def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    validate_image_dir = args.validate_image_dir
    validate_label_dir = args.validate_label_dir

    batch_size = 6
    num_workers = 4
    thresh = 0.1

    part_of_recall = partial(recall, thresh=thresh)
    part_of_precision = partial(precision, thresh=thresh)
    part_of_fbeta_score= partial(fbeta_score, thresh=thresh)

    model = fracnet(1, 1, 16)
    
    
    
    
    model = nn.DataParallel(model.cuda())
    optimizer = optim.Adam
    criterion = MixLoss(GHMLoss(), 0.2, DiceLoss(), 0.8)

    transforms = [tsfm.win(-200, 1000),tsfm.norm(-200, 1000)]
    rawdata_train = dataset_train(train_image_dir, train_label_dir,transform=transforms)
    data_train = dataset_train.get_dataloader(rawdata_train, batch_size, False,num_workers)
    rawdata_val = dataset_train(validate_image_dir, validate_label_dir,transform=transforms)
    data_val = dataset_train.get_dataloader(rawdata_val, batch_size, False,num_workers)

    databunch = DataBunch(data_train, data_val,collate_fn=dataset_train._collate_fn)

    start_learn = Learner(databunch,model,opt_func=optimizer,loss_func=criterion,metrics=[dice, part_of_recall, part_of_precision, part_of_fbeta_score])

    start_learn.fit_one_cycle(200,1e-3,pct_start=0,div_factor=1000,callbacks=[SaveModelCallback(start_learn),ShowGraph(start_learn)])

    if args.if_save:
        torch.save(model.module.state_dict(), "./fracnet_model.pth")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--train_label_dir", required=True)
    parser.add_argument("--validate_image_dir", required=True)
    parser.add_argument("--validate_label_dir", required=True)
    parser.add_argument("--if_save", default=False)
    args = parser.parse_args()
    main(args)
