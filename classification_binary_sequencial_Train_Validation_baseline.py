import os
import sys
import random

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

#CUDA_LAUNCH_BLOCKING=1
parent_dir = os.path.abspath('./')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import ( stratified_split,
                    only_train_classification_binary,
                    only_validation_classification_binary)

# Get the home directory
home_directory = os.path.expanduser('~')

sys.path.append("/media/ashish/T7 Shield/OS_prediction")

print(str(datetime.now()), ': Process Started')
clinical_data_date_cf = "2022-09-16"    #training data; clinical data date
clinical_data_date = pd.to_datetime(clinical_data_date_cf, format="%Y-%m-%d") 
print(str(datetime.now()), ": clinical_data_date: ", clinical_data_date)

from models.ResNet import ResNet50

from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse

def file_exists(file_path):
    return os.path.exists(file_path)

parser = argparse.ArgumentParser()
parser.add_argument('--fold', dest='fold', type=str, help='fold number')
parser.add_argument('--max_epochs', dest='max_epochs', type=str, help='max epoch number')
parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='dataset path')
parser.add_argument('--channel_flag', dest='channel_flag', type=str, help='channel description')
parser.add_argument('--model_name', dest='model_name', type=str, help='model name')
parser.add_argument('--exp_seed', dest='exp_seed', type=str, help='exp_seed')
parser.add_argument('--augmentation', dest='augmentation', type=str, help='data augmentation')
parser.add_argument('--year', dest='year', type=str, help='year for overall survival')


print("--------------Experiment without time information--------------")

param_args = parser.parse_args()

# Retrieve arguments
fold = int(param_args.fold) #int(sys.argv[1])
max_epochs = int(param_args.max_epochs) #int(sys.argv[2])
dataset_path = param_args.dataset_path
channel_flag = param_args.channel_flag
model_name = param_args.model_name
exp_seed = param_args.exp_seed
augmentation = param_args.augmentation
year = float(param_args.year)

print("arguments: ")
print("fold: ", fold)
print("maxepoch: ", max_epochs)
print("dataset_path: ", dataset_path)
print("channel_flag: ", channel_flag)
print("model_name: ", model_name)
print("exp_seed: ", exp_seed)
print("augmentation: ", augmentation)
print("year: ", year)

if augmentation == "True":
    from generate_dataset.generate_dataset_binary_with_augment import prepare_data_binary
elif augmentation == "False":
    from generate_dataset.generate_dataset_binary import prepare_data_binary
else:
    print("Data augmentation value is incorrect")
    sys.exit()

outcome = "Overall_Survival"
k_fold = 5
learning_rate = 1e-4 
weight_decay = 1e-5

args = {"num_workers": 4,
        "batch_size_val": 1,
        "batch_size_pred": 22}
number_of_timeframes = 6
oversampling_factor = 2

if model_name == "ResNet-50":
    batch_size_train = 10
else:
    print("Model name is incorrect")
    sys.exit()

df_path = "/media/ashish/T7 Shield/OS_prediction/DataFrame/Cross_Validation/LC_dataset_for_model_training.xlsx"

df = pd.read_excel(df_path)
mask = df["SUV_MIP"].apply(file_exists)
df = df[mask].reset_index(drop=True)
print("Final shape of dataframe is: ", df.shape)
print("sample path: ", df["SUV_MIP"].iloc[0])

classification_type = "Binary_Classification"

if outcome == "Overall_Survival":
    folder_name = "Overall_Survival"
    output_channels = 1
else:
    print("Outcome value is incorrect")
    sys.exit()
    
base_path_output = "/media/ashish/T7 Shield/OS_prediction/Output/Cross_Validation/Classification/" + folder_name  + "/Binary_Classification/" + model_name

if channel_flag=="SUV_CT_T_12":
    input_channels = 12

    if year==0.5:
        survival_days_clip = 180
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/0.5_year/seed_{exp_seed}/"
    elif year==1:
        survival_days_clip = 360
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/1_year/seed_{exp_seed}/"
    elif year==1.5:
        survival_days_clip = 540
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/1.5_year/seed_{exp_seed}/"
    elif year==2:
        survival_days_clip = 720
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/2_year/seed_{exp_seed}/"
    elif year==2.5:
        survival_days_clip = 900
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/2.5_year/seed_{exp_seed}/"
    elif year==3:
        survival_days_clip = 1080
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/3_year/seed_{exp_seed}/"
    elif year==3.5:
        survival_days_clip = 1260
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/3.5_year/seed_{exp_seed}/"
    elif year==4:
        survival_days_clip = 1440
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/4_year/seed_{exp_seed}/"
    elif year==4.5:
        survival_days_clip = 1620
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/4.5_year/seed_{exp_seed}/"
    elif year==5:
        survival_days_clip = 1800
        path_output = base_path_output + f"/SUV_CT_T_12_Channels/baseline/5_year/seed_{exp_seed}/"
    else:
        print("Year not provided")
        sys.exit()
else:
    print("Output path not provided")
    sys.exit()  

print("Input Channels: ", input_channels)

print("path_output: ", path_output)

if not os.path.exists(path_output):
    os.makedirs(path_output)

pre_trained_weights = False
checkpoint_path = ""

### K-Fold Cross Validation
for k in tqdm(range(k_fold)):
    if k == fold:
        print("\nCross Validation for fold: {}".format(k))
  
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        metric_values = {"AUC": [], "Avg_Precision": []}
        best_val_loss = 999999

        print("Network Initialization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        if model_name == "ResNet-50":
            model = ResNet50(num_classes=output_channels, channels=input_channels).to(device)
        else:
            print("Model not found")
            sys.exit(1)

        if pre_trained_weights:
            # Load pre trained weights
            print("\nCheckpoint Loading for Cross Validation: {}".format(k))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["net"])
        else:
            print("\nTraining from Scratch!!")
            epoch_to_continue = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if not os.path.exists(path_output + "CV_" + str(k) + '/Network_Weights/'):
            os.makedirs(path_output + "CV_" + str(k) + '/Network_Weights/')

        if not os.path.exists(path_output + "CV_" + str(k) + '/Metrics/'):
            os.makedirs(path_output + "CV_" + str(k) + '/Metrics/')
        
        if not os.path.exists(path_output + "CV_" + str(k) + '/MIPs/'):
            os.makedirs(path_output + "CV_" + str(k) + '/MIPs/')

        if not os.path.exists(path_output + "CV_" + str(k) + '/Dynamic_DataFrames/'):
            os.makedirs(path_output + "CV_" + str(k) + '/Dynamic_DataFrames/')

        loss_function = torch.nn.BCEWithLogitsLoss()

        train_loss = []
        val_loss = []
        epochs_list_train = []
        epochs_list_val = []


        # Reduce learning rate by 10% on plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Function to get the current learning rate
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']  # Assuming all groups share the same LR

        df["clinical_data_date"] = clinical_data_date
        df["scan_to_clinical_data_date_days"] = df.apply(lambda x: abs((clinical_data_date - x["scan_date"])/np.timedelta64(1, 'D')), axis=1)

        df_train, df_val = stratified_split(df, k_fold, k, outcome)

        df_train["Overall_Survival"] = df_train.apply(lambda x: 0 if x["survival_days"]<=survival_days_clip else 1, axis=1)
        condition = np.logical_or(df_train["Overall_Survival"]==0, np.logical_and(df_train["Overall_Survival"]==1, df_train["scan_to_clinical_data_date_days"]>=survival_days_clip))
        df_train = df_train[condition]
        df_train.reset_index(drop=True, inplace=True)

        df_val["Overall_Survival"] = df_val.apply(lambda x: 0 if x["survival_days"]<=survival_days_clip else 1, axis=1)
        condition = np.logical_or(df_val["Overall_Survival]==0, np.logical_and(df_val["Overall_Survival"]==1, df_val["scan_to_clinical_data_date_days"]>=survival_days_clip))
        df_val = df_val[condition]
        df_val.reset_index(drop=True, inplace=True)

        print("\nNumber of exams in Training set: ", len(df_train))
        print("\nNumber of exams in Validation set: ", len(df_val))

        print("\nPatient's " + outcome + " distribution in Training set: ", df_train.groupby(outcome)['patient_ID'].nunique())
        print("\nPatient's " + outcome + " distribution in Validation set: ", df_val.groupby(outcome)['patient_ID'].nunique())


        if pre_trained_weights:
            epoch_to_continue = 0 #best_metric_epoch + 1
        else:
            epoch_to_continue = 0

        lr_list = []

        for epoch in tqdm(range(max_epochs)):
            epoch += epoch_to_continue

            train_files, train_loader = prepare_data_binary(args, df_train, batch_size_train, shuffle=True, label=outcome, channel_flag=channel_flag)

            epoch_loss, train_loss = only_train_classification_binary(model, epoch, k, path_output, train_loader, optimizer, loss_function, device, train_loss, outcome)

            print(f"\nTraining epoch {epoch} average loss: {epoch_loss:.4f}")
            epochs_list_train.append(epoch + 1)

            np.save(os.path.join(path_output, "CV_" + str(k) + "/Training_Loss.npy"), train_loss)

            train_loss_path = os.path.join(path_output, "CV_" + str(k), "Epoch_vs_Training_Loss.jpg")

            if len(train_loss) >= 2:
                # Plot and save figures
                plt.figure(1)
                plt.plot(epochs_list_train, train_loss)
                plt.xlabel('Number of Epochs')
                plt.ylabel('Training_loss')
                plt.savefig(train_loss_path, dpi=400) 
                plt.clf()  # Clear the loss figure for next update


            epochs_list_val.append(epoch + 1)
            metric_values, val_epoch_loss, val_loss, best_val_loss_new = only_validation_classification_binary(args, k, epoch, optimizer, loss_function, model, df_val, device, metric_values, path_output, best_val_loss, val_loss, outcome, channel_flag)
            best_val_loss = best_val_loss_new

            # Print updated learning rate
            current_lr = get_lr(optimizer)

            print(f"Current LR: {current_lr}")

            lr_list.append(current_lr)

            lr_path = os.path.join(path_output, "CV_" + str(k), "Epoch_vs_LR.jpg")

            if len(lr_list) >= 2:
                plt.figure(1)
                plt.plot(epochs_list_val, lr_list)
                plt.xlabel('Number of Epochs')
                plt.ylabel('LR')
                plt.savefig(lr_path, dpi=400)  
                plt.clf()  # Clear the loss figure for next update     

            scheduler.step(val_epoch_loss)  # Adjust LR based on validation loss            

            np.save(os.path.join(path_output, "CV_" + str(k) + "/LR.npy"), lr_list)
                
            print(f"\nValidation epoch {epoch} average loss: {val_epoch_loss:.4f}")

            np.save(os.path.join(path_output, "CV_" + str(k) + "/Validation_Loss.npy"), val_loss)
            
            np.save(os.path.join(path_output, "CV_" + str(k) + "/AUC.npy"), metric_values["AUC"])
            np.save(os.path.join(path_output, "CV_" + str(k) + "/Avg_Precision.npy"), metric_values["Avg_Precision"])

            val_loss_path = os.path.join(path_output, "CV_" + str(k), "Epoch_vs_Val_Loss.jpg")

            if len(val_loss) >= 2:
                plt.figure(1)
                plt.plot(epochs_list_val, val_loss)
                plt.xlabel('Number of Epochs')
                plt.ylabel('Validation_loss')
                plt.savefig(val_loss_path, dpi=400)  
                plt.clf()  # Clear the loss figure for next update      
