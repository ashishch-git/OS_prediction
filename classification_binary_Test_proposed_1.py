import os
import sys
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

#CUDA_LAUNCH_BLOCKING=1
parent_dir = os.path.abspath('./')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import ( create_allsurvivaldays_df,
                    create_test_allsurvivaldays_df,
                    testing_classification_binary_withTime)

# Get the home directory
home_directory = os.path.expanduser('~')

# Append the parent directory to the Python path
sys.path.append("/media/ashish/T7 Shield/OS_prediction")

clinical_data_date_cf = '2024-05-23'    #testing data; clinical data date
clinical_data_date = pd.to_datetime(clinical_data_date_cf, format="%Y-%m-%d") 
print("clinical_data_date: ", clinical_data_date)

from models.ResNet_withTime import ResNet50_withTime

import argparse
print(str(datetime.now()), ': Process Started')
parser = argparse.ArgumentParser()
parser.add_argument('--fold', dest='fold', type=str, help='fold number')
parser.add_argument('--channel_flag', dest='channel_flag', type=str, help='channel description')
parser.add_argument('--model_name', dest='model_name', type=str, help='model name')
parser.add_argument('--exp_seed', dest='exp_seed', type=str, help='exp_seed')
param_args = parser.parse_args()

# Retrieve arguments
fold = int(param_args.fold) #int(sys.argv[1])
channel_flag = param_args.channel_flag
model_name = param_args.model_name
exp_seed = param_args.exp_seed

print("arguments: ")
print("fold: ", fold)
print("channel_flag: ", channel_flag)
print("model_name: ", model_name)
print("exp_seed: ", exp_seed)

outcome = "Overall_Survival"
max_epochs = 300
k_fold = 5
learning_rate = 1e-4
weight_decay = 1e-5
args = {"num_workers": 4,
        "batch_size_val": 1,
        "batch_size_pred": 22}

df_path = "/media/ashish/T7 Shield/OS_prediction/DataFrame/Testing/LC_dataset_for_model_testing.xlsx"

df = pd.read_excel(df_path)

classification_type = "Binary_Classification"

if outcome == "Overall_Survival":
    folder_name = "Overall_Survival"
    output_channels = 1
else:
    print("Outcome value is incorrect")
    sys.exit()

base_path_output = "/media/ashish/T7 Shield/OS_prediction/Output/Testing/Classification/" + folder_name  + "/Binary_Classification/" + model_name

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

pre_trained_weights = True

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
        
        if not os.path.exists(path_output + "CV_" + str(k) + '/Testset_Metrics/'):
            os.makedirs(path_output + "CV_" + str(k) + '/Testset_Metrics/')

        loss_function = torch.nn.BCEWithLogitsLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_loss = []
        val_loss = []
        epochs_list_train = []
        epochs_list_val = []

        survival_days_list = list(range(30, (30 * 12 * 5) + 30, 30))

        df_val = df.copy()

        print("\nNumber of exams in Validation set: ", len(df_val))

        print("\nPatient's " + outcome + " distribution in Validation set: ", df_val.groupby(outcome)['patient_ID'].nunique())

        if pre_trained_weights:
            # Load pre trained weights
            print("\nCheckpoint Loading for Cross Validation: {}".format(k))
            for item in os.listdir(path_output + "CV_" + str(k) + '/Network_Weights/'):
                checkpoint_path = path_output + "CV_" + str(k) + '/Network_Weights/' + item
                print("checkpoint_path: ", checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["net"])
                break
        else:
            print("\nTraining from Scratch!!")
        
        csv_columns = ["unique_pat_ID_scan_date_days_period", "unique_pat_ID_scan_date", "sex", "patient_ID", "consentDate", "scan_date", "survival_days", "survival_months", "progression_free_months", "progression_free_days", "deceased", "deceasedDate", "diagnosisGroups", "DiaDate", "translatedDiagnosis", "RELAPSE1", "date", "RELAPSE2", "date 2", "days_period" , "limit_date", "clinical_data_date", outcome]
        

        print("\nGenerating validation data with dynamic timeframe")

        df_val_dt = create_test_allsurvivaldays_df(df_val, clinical_data_date, outcome, survival_days_list)

        df_val_dt["unique_pat_ID_scan_date_days_period"] = df_val_dt.apply(lambda x: str(x["patient_ID"]) + "_" + str(x["scan_date"])  + "_" + str(x["days_period"]), axis=1)
        df_val_dt["clinical_data_date"] = clinical_data_date

        df_val_dt2 = df_val_dt[csv_columns].copy()
        df_val_dt2.to_csv(os.path.join(path_output, "CV_" + str(k), "Testset_Metrics_withForecast", "df_test_dt.csv"), index=False)

        print("\nNumber of exams in Validation set: ", len(df_val_dt))
        print("\n" + outcome + " distribution Unique Patients in Validation set: ", df_val_dt.groupby(outcome)['patient_ID'].nunique())
        print("\n" + outcome + " distribution in Validation set: ", df_val_dt.groupby(outcome)['unique_pat_ID_scan_date_days_period'].nunique())

        test_loss = testing_classification_binary_withTime(args, k, optimizer, loss_function, model, df_val_dt, device, path_output, outcome, channel_flag)
            
        print(f"\nTesting average loss: {test_loss:.4f}")
