import torch
from torch import nn
import torch
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    RandCoarseDropoutd,
    RandGaussianSmoothd,
    RandAffined,
    RandHistogramShiftd,
    RandRotated,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensity,
    RandSpatialCropSamplesd,
    ScaleIntensityRangePercentilesd,
    ThresholdIntensityd#,
    #ForegroundMaskd
)
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import ToTensor
from monai.transforms.intensity.array import ScaleIntensity
from monai.transforms import SpatialCrop, RandRotate
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, train_files, label_files, transform):
        self.files = train_files
        
        self.labels = label_files

        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        files_path = self.files[index]
        
        labels = self.labels[index]

        files_new = self.transform(files_path)

        return files_new["PET_CT"], labels
    
def prepare_data_binary(args, df_train, batch_size, shuffle=None, label=None, channel_flag="SUV_CT_T_13"):
    if shuffle==True:
        df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)
    else:
        df_train_shuffled = df_train
    
    SUV_MIP_train = df_train_shuffled["SUV_MIP"].tolist()
    SUV_bone_train = df_train_shuffled["SUV_bone"].tolist()
    SUV_lean_train = df_train_shuffled["SUV_lean"].tolist()
    SUV_adipose_train = df_train_shuffled["SUV_adipose"].tolist()
    SUV_air_train = df_train_shuffled["SUV_air"].tolist()
    CT_MeanIP_train = df_train_shuffled["CT_MeanIP"].tolist()
    CT_bone_train = df_train_shuffled["CT_bone"].tolist()
    CT_lean_train = df_train_shuffled["CT_lean"].tolist()
    CT_adipose_train = df_train_shuffled["CT_adipose"].tolist()
    CT_air_train = df_train_shuffled["CT_air"].tolist()
    SUV_TSeg_MIP_train = df_train_shuffled["SUV_TSeg_MIP"].tolist()
    CT_TSeg_MeanIP_train = df_train_shuffled["CT_TSeg_MeanIP"].tolist()
    
    if label == "Overall_Survival":
        label_train = df_train_shuffled["Overall_Survival"].tolist()
    else:
        print("Label is incorrect")
        sys.exit()
     
    train_files = [
        {"SUV_MIP": SUV_MIP_name, "SUV_bone": SUV_bone_name, "SUV_lean": SUV_lean_name, "SUV_adipose": SUV_adipose_name, "SUV_air": SUV_air_name, 
         "CT_MeanIP": CT_MeanIP_name, "CT_bone": CT_bone_name, "CT_lean": CT_lean_name, "CT_adipose": CT_adipose_name, "CT_air": CT_air_name, "SUV_TSeg_MIP": SUV_TSeg_MIP_name, "CT_TSeg_MeanIP": CT_TSeg_MeanIP_name}
         for SUV_MIP_name, SUV_bone_name, SUV_lean_name, SUV_adipose_name, SUV_air_name, CT_MeanIP_name, CT_bone_name, CT_lean_name, CT_adipose_name, CT_air_name, SUV_TSeg_MIP_name, CT_TSeg_MeanIP_name, labels in
         zip(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MeanIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, SUV_TSeg_MIP_train, CT_TSeg_MeanIP_train, label_train)
    ]

    label_files = [labels for labels in label_train]

    train_keys = ["SUV_MIP", "SUV_bone", "SUV_lean", "SUV_adipose", "SUV_air", "CT_MeanIP", "CT_bone", "CT_lean", "CT_adipose", "CT_air", "SUV_TSeg_MIP", "CT_TSeg_MeanIP"]

    train_transforms = Compose(
        [
            LoadImaged(keys = train_keys),
            
            EnsureChannelFirstd(keys = train_keys),

            RandRotated(keys = train_keys, prob=0.1, range_x=[-0.015, 0.015]),
            RandGaussianSmoothd(keys = train_keys, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), prob=0.1, approx='erf'),
            RandAffined(keys = train_keys, prob=0.1, translate_range=(30,30)),          

            ConcatItemsd(keys=train_keys, name="PET_CT", dim=0),# concatenate pet and ct channels
            ToTensord(keys=["PET_CT"]),
        ]
    )

    train_ds = ImageDataset(train_files, label_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=args["num_workers"])
         
    return train_files, train_loader
