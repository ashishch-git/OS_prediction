import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import ToTensor
from monai.transforms.intensity.array import ScaleIntensity

class ImageDataset(Dataset):
    def __init__(self, SUV_MIP_files, SUV_bone_files, SUV_lean_files, SUV_adipose_files, SUV_air_files, CT_MeanIP_files, CT_bone_files, CT_lean_files, CT_adipose_files, CT_air_files, SUV_TSeg_MIP_files, CT_TSeg_MeanIP_files, labels, channel_flag):
        self.SUV_MIP_files = SUV_MIP_files
        self.SUV_bone_files = SUV_bone_files
        self.SUV_lean_files = SUV_lean_files
        self.SUV_adipose_files = SUV_adipose_files
        self.SUV_air_files = SUV_air_files
        self.CT_MeanIP_files = CT_MeanIP_files
        self.CT_bone_files = CT_bone_files
        self.CT_lean_files = CT_lean_files
        self.CT_adipose_files = CT_adipose_files
        self.CT_air_files = CT_air_files
        self.SUV_TSeg_MIP_files = SUV_TSeg_MIP_files
        self.CT_TSeg_MeanIP_files = CT_TSeg_MeanIP_files

        self.labels = labels

        self.channel_flag = channel_flag
 
        self.transform = Compose(
            [
                LoadImage(image_only=True, dtype=float),
                ToTensor(),
                ScaleIntensity(minv=0, maxv=1)
            ]
        )
    
    def __len__(self):
        return len(self.SUV_MIP_files)
    
    def __getitem__(self, index):
        SUV_MIP_path = self.SUV_MIP_files[index]
        SUV_bone_path = self.SUV_bone_files[index]
        SUV_lean_path = self.SUV_lean_files[index]
        SUV_adipose_path = self.SUV_adipose_files[index]
        SUV_air_path = self.SUV_air_files[index]
        CT_MeanIP_path = self.CT_MeanIP_files[index]
        CT_bone_path = self.CT_bone_files[index]
        CT_lean_path = self.CT_lean_files[index]
        CT_adipose_path = self.CT_adipose_files[index]
        CT_air_path = self.CT_air_files[index]
        SUV_TSeg_MIP_path = self.SUV_TSeg_MIP_files[index]
        CT_TSeg_MeanIP_path = self.CT_TSeg_MeanIP_files[index]

        labels = self.labels[index]

        # Load and transform the images
        SUV_MIP = self.transform(SUV_MIP_path)
        SUV_bone = self.transform(SUV_bone_path)
        SUV_lean = self.transform(SUV_lean_path)
        SUV_adipose = self.transform(SUV_adipose_path)
        SUV_air = self.transform(SUV_air_path)
        CT_MeanIP = self.transform(CT_MeanIP_path)
        CT_bone = self.transform(CT_bone_path)
        CT_lean = self.transform(CT_lean_path)
        CT_adipose = self.transform(CT_adipose_path)
        CT_air = self.transform(CT_air_path)
        SUV_TSeg_MIP = self.transform(SUV_TSeg_MIP_path)
        CT_TSeg_MeanIP = self.transform(CT_TSeg_MeanIP_path)

        # Concatenate the images along the channel dimension
        SUV_MIP_new = torch.unsqueeze(SUV_MIP, 0)
        SUV_bone_new = torch.unsqueeze(SUV_bone, 0)
        SUV_lean_new = torch.unsqueeze(SUV_lean, 0)
        SUV_adipose_new = torch.unsqueeze(SUV_adipose, 0)
        SUV_air_new = torch.unsqueeze(SUV_air, 0)
        CT_MeanIP_new = torch.unsqueeze(CT_MeanIP, 0)
        CT_bone_new = torch.unsqueeze(CT_bone, 0)
        CT_lean_new = torch.unsqueeze(CT_lean, 0)
        CT_adipose_new = torch.unsqueeze(CT_adipose, 0)
        CT_air_new = torch.unsqueeze(CT_air, 0)
        SUV_TSeg_MIP_new = torch.unsqueeze(SUV_TSeg_MIP, 0)
        CT_TSeg_MeanIP_new = torch.unsqueeze(CT_TSeg_MeanIP, 0)
        
        if self.channel_flag=="SUV_CT_T_12":
            multi_channel_input = torch.cat((SUV_MIP_new, SUV_bone_new, SUV_lean_new, SUV_adipose_new, SUV_air_new, CT_MeanIP_new, CT_bone_new, CT_lean_new, CT_adipose_new, CT_air_new, SUV_TSeg_MIP_new, CT_TSeg_MeanIP_new), dim=0)
        else:
            print("Channel flag is incorrect")
            sys.exit()
            
        return multi_channel_input, labels
    
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
         "CT_MeanIP": CT_MeanIP_name, "CT_bone": CT_bone_name, "CT_lean": CT_lean_name, "CT_adipose": CT_adipose_name, "CT_air": CT_air_name, "SUV_TSeg_MIP": SUV_TSeg_MIP_name, "CT_TSeg_MeanIP": CT_TSeg_MeanIP_name, "labels": labels}
         for SUV_MIP_name, SUV_bone_name, SUV_lean_name, SUV_adipose_name, SUV_air_name, CT_MeanIP_name, CT_bone_name, CT_lean_name, CT_adipose_name, CT_air_name, SUV_TSeg_MIP_name, CT_TSeg_MeanIP_name, labels in
         zip(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MeanIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, SUV_TSeg_MIP_train, CT_TSeg_MeanIP_train, label_train)
    ]

    train_ds = ImageDataset(SUV_MIP_train, SUV_bone_train, SUV_lean_train, SUV_adipose_train, SUV_air_train, CT_MeanIP_train, CT_bone_train, CT_lean_train, CT_adipose_train, CT_air_train, SUV_TSeg_MIP_train, CT_TSeg_MeanIP_train, label_train, channel_flag)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=args["num_workers"])

    return train_files, train_loader
