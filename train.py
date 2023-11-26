import glob
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
from monai.config import print_config
from monai.data import (
ArrayDataset,
create_test_image_3d,
decollate_batch,
DataLoader,
CacheDataset
)
from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DeepSupervisionLoss, DiceLoss
from monai.metrics import compute_dice, DiceMetric
from monai.networks.nets import UNet, SegResNet, SegResNetDS, SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    Resized,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandAffined,
    RandFlipd,
    ToTensord,

)
from monai.utils.misc import ImageMetaKey
from monai.utils import first
import ignite
import torch
import random
import pandas as pd
import shutil


metrics_dir = '/WAVE/users/unix/smalladi/varian_ml/metrics/10-26_300/'


def read_files(resampled_ct_path, resampled_pt_path, resampled_label_path):

    train_images = sorted(
        glob.glob(os.path.join(resampled_ct_path, "*_CT*")))
    train_images2 = sorted(
        glob.glob(os.path.join(resampled_pt_path, "*_PT*")))
    train_labels = sorted(
        glob.glob(os.path.join(resampled_label_path, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "image2": pet_image, 'label': label_name}
        for image_name, pet_image, label_name in zip(train_images, train_images2, train_labels)
    ]
    # temp = [{'image': '/WAVE/users/unix/smalladi/varian_ml/hecktor2022_training/hecktor2022/resampled_largerCt/MDA-184__CT_nobrain.nii.gz', 'image2': '/WAVE/users/unix/smalladi/varian_ml/hecktor2022_training/hecktor2022/resampled_largerPt/MDA-184__PT_nobrain.nii.gz', 'label':'/WAVE/users/unix/smalladi/varian_ml/hecktor2022_training/hecktor2022/resampled_largerlabel/MDA-184__CT_nobrain.nii.gz'}]
    # data_dicts.remove(temp[0])
    x=[i for i in range(524)]
    #random.shuffle(x)
    train_index,val_index,test_index=x[:450],x[450:520],x[520:]
    train_files=[]
    val_files=[]
    test_files=[]
    for i in train_index:
        train_files.append(data_dicts[i])
    for i in val_index:
        val_files.append(data_dicts[i])
    for i in test_index:
        test_files.append(data_dicts[i])
    return train_files, val_files, test_files


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return (model,
            optimizer,
            checkpoint['epoch'],
            checkpoint['best_metric'],
            checkpoint['train_time'],
            checkpoint['epoch_loss_values'],
            checkpoint['metric_values'],
            checkpoint['metric_values_1'],
            checkpoint['metric_values_2']
            )

def save_ckp(state, is_best, metrics_dir):
    f_path = os.path.join(metrics_dir, 'checkpoint.pt')
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(metrics_dir, 'best_model.pt')
        shutil.copyfile(f_path, best_fpath)

root_dir = '/WAVE/users/unix/smalladi/varian_ml/Hecktor22'
data_dir = 'hecktor2022_training/hecktor2022'
resampled_ct_path = '/WAVE/users/unix/smalladi/varian_ml/hecktor2022_training/resampled/imagesTr'
resampled_pt_path = '/WAVE/users/unix/smalladi/varian_ml/hecktor2022_training/resampled/imagesTr'
resampled_label_path = '/WAVE/users/unix/smalladi/varian_ml/hecktor2022_training/resampled'


train_files, val_files, test_files = read_files(resampled_ct_path, resampled_pt_path, resampled_label_path)

ct_a_min = -200
ct_a_max = 400
pt_a_min = 0
pt_a_max = 25
crop_samples = 2
input_size = [96, 96, 96]
modes_2d = ['bilinear', 'bilinear', 'nearest']
p = 0.5
strength = 1
image_keys = ["image", "image2", "label"]

train_transforms = Compose([
    LoadImaged(keys=["image", "image2", "label"]),
    EnsureChannelFirstd(keys = ["image", "image2", "label"]),
    Orientationd(keys=["image", "image2", "label"], axcodes="RAS"),
    # Spacingd(
    #     keys=image_keys,
    #     pixdim=(1, 1, 1),
    #     mode=modes_2d,
    # ),
    ScaleIntensityRanged(keys=['image'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['image2'], a_min=pt_a_min, a_max=pt_a_max, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=image_keys, source_key='image'),
    RandCropByPosNegLabeld(
        keys=image_keys,
        label_key='label',
        spatial_size=input_size,
        pos=1,
        neg=1,
        num_samples=crop_samples,
        image_key='image',
        image_threshold=0,
    ),
    RandAffined(keys=image_keys, prob=p,
                    translate_range=(round(10 * strength), round(10 * strength), round(10 * strength)),
                    padding_mode='border', mode=modes_2d),
    RandAffined(keys=image_keys, prob=p, scale_range=(0.10 * strength, 0.10 * strength, 0.10 * strength),
                    padding_mode='border', mode=modes_2d),
    RandFlipd(keys=["image", "image2", "label"], prob=p/3, spatial_axis=0),
    RandFlipd(keys=["image", "image2", "label"], prob=p/3, spatial_axis=1),
    RandFlipd(keys=["image", "image2", "label"], prob=p/3, spatial_axis=2),
    RandShiftIntensityd(
            keys=["image", "image2"],
            offsets=0.10,
            prob=p,
        ),
    ToTensord(keys=["image", "image2", "label"])
])
val_transforms = Compose([
    LoadImaged(keys=["image", "image2", "label"]),
    EnsureChannelFirstd(keys = ["image", "image2", "label"]),
    Orientationd(keys=["image", "image2", "label"], axcodes="RAS"),
    # Spacingd(
    #     keys=image_keys,
    #     pixdim=(1, 1, 1),
    #     mode=modes_2d,
    # ),
    ScaleIntensityRanged(keys=['image'], a_min=ct_a_min, a_max=ct_a_max, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['image2'], a_min=pt_a_min, a_max=pt_a_max, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=image_keys, source_key='image'),
    ToTensord(keys=["image", "image2", "label"])
])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=0.0)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)


VAL_AMP = True
n_classes = 3
n_channels = 2
input_size = (96, 96, 96)
max_epochs = 300
device = torch.device("cuda:0")
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    init_filters=16,
    blocks_up=[1, 1, 1],
    in_channels = n_channels,
    out_channels= n_classes,
    dropout_prob = 0.2
).to(device)


# model = SegResNetDS(
#   init_filters = 32,
#   blocks_down = [1, 2, 2, 4, 4, 4],
#   norm = 'BATCH',
#   in_channels = n_channels,
#   out_channels = n_classes,
#   dsdepth = 4
# ).to(device)

loss_function = DiceLoss(softmax=True, to_onehot_y=True)
# loss_function = DeepSupervisionLoss(loss_function)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002,  weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
post_label = AsDiscrete(to_onehot=n_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=n_classes)

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True
start_epoch = 0
train_time = 0


roi_size = (192, 192, 192)
sw_batch_size = 2
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
metric_values_1 = list()
metric_values_2 = list()
best_metrics_epochs_and_time = [[], [], []]
total_start = time.time()

# ckp_path = os.path.join(metrics_dir, 'checkpoint.pt')
# (
#     model,
#     optimizer,
#     start_epoch,
#     best_metric,
#     train_time,
#     epoch_loss_values,
#     metric_values,
#     metric_values_1,
#     metric_values_2
#     ) = load_ckp(ckp_path, model, optimizer)

for epoch in range(start_epoch, max_epochs):
    is_best = False
    epoch_start = time.time()
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + 1, max_epochs))
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputct, inputpt, labels = (
            batch_data['image'].to(device),
            batch_data['image2'].to(device), 
            batch_data['label'].to(device)
        )
        inputs = torch.concat([inputct, inputpt], axis=1)
        optimizer.zero_grad()
        # try:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        # except Exception as e:
        #     print(labels.meta[ImageMetaKey.FILENAME_OR_OBJ])
        #     print(e)
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputct, val_inputpt, val_label = (
                val_data["image"].to(device),
                val_data["image2"].to(device),
                val_data["label"].to(device),
            )
            val_inputs = torch.concat([val_inputct, val_inputpt], axis=1)
            # try:
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            # except Exception as e:
                # print(labels.meta[ImageMetaKey.FILENAME_OR_OBJ])
                # print(e)
            val_label_list = decollate_batch(val_label)
            val_label_convert = [post_label(val_label_tensor) for val_label_tensor in val_label_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_label_convert)
            dice_metric_batch(y_pred=val_output_convert, y=val_label_convert) 
        metric = dice_metric.aggregate().item()
        metric_values.append(metric)
        metric_batch = dice_metric_batch.aggregate()
        metric_1 = metric_batch[0].item()
        metric_values_1.append(metric_1)
        metric_2 = metric_batch[1].item()
        metric_values_2.append(metric_2)
        dice_metric.reset()
        dice_metric_batch.reset()

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            best_metrics_epochs_and_time[0].append(best_metric)
            best_metrics_epochs_and_time[1].append(best_metric_epoch)
            best_metrics_epochs_and_time[2].append(time.time() - total_start)
            is_best = True
            print("saved new best metric model")
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f" 1: {metric_1:.4f} 2: {metric_2:.4f}"
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
    train_time += (time.time() - epoch_start)
    checkpoint = {
        'epoch': epoch + 1,
        'best_metric': best_metric,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_time': train_time,
        'epoch_loss_values': epoch_loss_values,
        'metric_values': metric_values,
        'metric_values_1': metric_values_1,
        'metric_values_2': metric_values_2
    }

    save_ckp(checkpoint, is_best, metrics_dir)
    print(f"Epoch {epoch + 1} completed")
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

best_metrics_epochs_and_time_df = pd.DataFrame(best_metrics_epochs_and_time)
epoch_loss_values_df = pd.DataFrame(epoch_loss_values)
metric_values_df = pd.DataFrame(metric_values)
metric_values_1_df = pd.DataFrame(metric_values_1)
metric_values_2_df = pd.DataFrame(metric_values_2)
best_metrics_epochs_and_time_df.to_csv(metrics_dir + "best_metrics_epochs_and_time.csv")
epoch_loss_values_df.to_csv(metrics_dir + "epoch_loss_values.csv")
metric_values_df.to_csv(metrics_dir + "metric_values.csv")
metric_values_1_df.to_csv(metrics_dir + "metric_values_1.csv")
metric_values_2_df.to_csv(metrics_dir + "metric_values_2.csv")

print("Data saved to metrics directory")