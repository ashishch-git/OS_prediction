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
                    create_traindynamictime_df,
                    create_allsurvivaldays_df,
                    only_train_classification_binary_withTime,
                    only_validation_classification_binary_withTime)

# Get the home directory
home_directory = os.path.expanduser('~')

sys.path.append("/media/ashish/T7 Shield/OS_prediction")

print(str(datetime.now()), ': Process Started')
clinical_data_date_cf = "2022-09-16"    #training data; clinical data date
clinical_data_date = pd.to_datetime(clinical_data_date_cf, format="%Y-%m-%d") 
print(str(datetime.now()), ": clinical_data_date: ", clinical_data_date)

from models.ResNet_withTime import ResNet50_withTime

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
parser.add_argument('--augmentation', dest='augmentation', type=str, help='data augmentation')
parser.add_argument('--exp_seed', dest='exp_seed', type=str, help='exp_seed')

print("--------------Experiment with time information--------------")

param_args = parser.parse_args()

# Retrieve arguments
fold = int(param_args.fold)
max_epochs = int(param_args.max_epochs)
dataset_path = param_args.dataset_path
channel_flag = param_args.channel_flag
model_name = param_args.model_name
exp_seed = param_args.exp_seed
augmentation = param_args.augmentation
    
print("arguments: ")
print("fold: ", fold)
print("maxepoch: ", max_epochs)
print("dataset_path: ", dataset_path)
print("channel_flag: ", channel_flag)
print("model_name: ", model_name)
print("augmentation: ", augmentation)
print("exp_seed: ", exp_seed)

if augmentation == "True":
    from generate_dataset.generate_dataset_binary_with_augment_withTime import prepare_data_binary_withTime
elif augmentation == "False":
    from generate_dataset.generate_dataset_binary_withTime import prepare_data_binary_withTime
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

classification_type = "Binary_Classification" #"Multi-label_Classification"

if outcome == "Overall_Survival":
    folder_name = "Overall_Survival"
    output_channels = 1
else:
    print("Outcome value is incorrect")
    sys.exit()

base_path_output = "/media/ashish/T7 Shield/OS_prediction/Output/Cross_Validation/Classification/" + folder_name  + "/Binary_Classification/" + model_name

if channel_flag=="SUV_CT_T_12":
    input_channels = 12
    path_output = base_path_output + f"/SUV_CT_T_12_Channels/proposed_1/seed_{exp_seed}/"
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
            model = ResNet50_withTime(num_classes=output_channels, channels=input_channels).to(device)
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

        lr_list = []

        # Reduce learning rate by 10% on plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Function to get the current learning rate
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']  # Assuming all groups share the same LR

        survival_days_list = list(range(30, (30 * 12 * 5) + 30, 30))

        df_train, df_val = stratified_split(df, k_fold, k, outcome)

        print("\nNumber of exams in Training set: ", len(df_train))
        print("\nNumber of exams in Validation set: ", len(df_val))

        print("\nPatient's " + outcome + " distribution in Training set: ", df_train.groupby(outcome)['patient_ID'].nunique())
        print("\nPatient's " + outcome + " distribution in Validation set: ", df_val.groupby(outcome)['patient_ID'].nunique())

        for epoch in tqdm(range(max_epochs)):
            epoch += epoch_to_continue

            csv_columns = ["unique_pat_ID_scan_date_days_period", "unique_pat_ID_scan_date", "patientDOB", "age", "sex", "inclusionYear", "patient_ID", "consentDate", "scan_date", "survival_days", "survival_months", "progression_free_months", "progression_free_days", "deceased", "deceasedDate", "diagnosisGroups", "DiaDate", "translatedDiagnosis", "indication", "RELAPSE1", "date", "RELAPSE2", "date 2", "days_period" , "limit_date", "clinical_data_date", outcome]

            df_train_dt1 = create_traindynamictime_df(df_train, clinical_data_date, outcome, arg_seed=epoch, number_of_timeframes=6, oversampling_factor=2)
            df_train2 = df_train[df_train.survival_days <= 40].copy()
            df_train_dt2 = create_traindynamictime_df(df_train2, clinical_data_date, outcome, arg_seed=epoch, number_of_timeframes=6, oversampling_factor=12, n_max=1000)
            df_train_dt = pd.concat([df_train_dt1, df_train_dt2], ignore_index=True)

            print("Combined Dynamic Training Dataset: ", df_train_dt.shape)

            df_train_dt["unique_pat_ID_scan_date_days_period"] = df_train_dt.apply(lambda x: str(x["patient_ID"]) + "_" + str(x["scan_date"])  + "_" + str(x["days_period"]), axis=1)
            df_train_dt["clinical_data_date"] = clinical_data_date

            
            df_train_dt2 = df_train_dt[csv_columns].copy()
            df_train_dt2.to_csv(os.path.join(path_output, "CV_" + str(k) + '/Dynamic_DataFrames', "epoch_" + str(epoch) + "_df_train_dt.csv"), index=False) 

            print("\nNumber of exams in Training set: ", len(df_train_dt))
            print("\n" + outcome + " distribution Unique Patients in Training set: ", df_train_dt.groupby(outcome)['patient_ID'].nunique())
            print("\n" + outcome + " distribution in Training set: ", df_train_dt.groupby(outcome)['unique_pat_ID_scan_date_days_period'].nunique())

            train_files, train_loader = prepare_data_binary_withTime(args, df_train_dt, batch_size_train, shuffle=True, label=outcome, channel_flag=channel_flag)

            epoch_loss, train_loss = only_train_classification_binary_withTime(model, epoch, k, path_output, train_loader, optimizer, loss_function, device, train_loss, outcome)
            print(f"\nTraining epoch {epoch} average loss: {epoch_loss:.4f}")
            epochs_list_train.append(epoch + 1)

            np.save(os.path.join(path_output, "CV_" + str(k) + "/Training_Loss.npy"), train_loss)

            train_loss_path = os.path.join(path_output, "CV_" + str(k), "Epoch_vs_Training_Loss.jpg")

            if len(train_loss) >= 2:
                plt.figure(1)
                plt.plot(epochs_list_train, train_loss)
                plt.xlabel('Number of Epochs')
                plt.ylabel('Training_loss')
                plt.savefig(train_loss_path, dpi=400) 
                plt.clf()  # Clear the loss figure for next update

            epochs_list_val.append(epoch + 1)
            print("\nGenerating validation data with dynamic timeframe")

            df_val_dt = create_allsurvivaldays_df(df_val, clinical_data_date, outcome, survival_days_list)
            df_val_dt["unique_pat_ID_scan_date_days_period"] = df_val_dt.apply(lambda x: str(x["patient_ID"]) + "_" + str(x["scan_date"])  + "_" + str(x["days_period"]), axis=1)
            df_val_dt["clinical_data_date"] = clinical_data_date

            if epoch == 0:
                df_val_dt2 = df_val_dt[csv_columns].copy()
                df_val_dt2.to_csv(os.path.join(path_output, "CV_" + str(k) + '/Dynamic_DataFrames', "epoch_" + str(epoch) + "_df_val_dt.csv"), index=False)

            print("\nNumber of exams in Validation set: ", len(df_val_dt))
            print("\n" + outcome + " distribution Unique Patients in Validation set: ", df_val_dt.groupby(outcome)['patient_ID'].nunique())
            print("\n" + outcome + " distribution in Validation set: ", df_val_dt.groupby(outcome)['unique_pat_ID_scan_date_days_period'].nunique())

            metric_values, val_epoch_loss, val_loss, best_val_loss_new = only_validation_classification_binary_withTime(args, k, epoch, optimizer, loss_function, model, df_val_dt, device, metric_values, path_output, best_val_loss, val_loss, outcome, channel_flag)
            
            best_val_loss = best_val_loss_new

            # Print updated learning rate
            current_lr = get_lr(optimizer)

            print(f"Current LR: {current_lr}")

            lr_list.append(current_lr)

            lr_path = os.path.join(path_output, "CV_" + str(k), "Epoch_vs_LR.jpg")

            if len(val_loss) >= 2:
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
