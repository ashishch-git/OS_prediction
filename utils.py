import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from re import S
from tqdm import tqdm

from scipy.ndimage.measurements import label
from sklearn import metrics
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, cohen_kappa_score, matthews_corrcoef, average_precision_score, precision_recall_curve, roc_auc_score

import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_auroc, multiclass_accuracy, multiclass_recall, multiclass_precision
from torchvision.ops.focal_loss import sigmoid_focal_loss

from generate_dataset.generate_dataset_binary import prepare_data_binary
from generate_dataset.generate_dataset_binary_withTime import prepare_data_binary_withTime

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

##### Helper functions
def working_system(system):
    if system == 1:
        pass        
    elif system == 2:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    else:
        print("Invalid system")

def focal_loss(logits, targets, loss_function):

    alpha = 0.25  # Class balance parameter
    gamma = 2.0 #1.0 #2.0  # Focusing parameter
    
    pt = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    
    loss = loss_function(logits, targets)
    
    focal_loss = alpha * (1 - pt) ** gamma * loss

    mean_focal_loss = torch.mean(focal_loss)

    return mean_focal_loss

def make_dirs(path, k):
    path_CV = os.path.join(path, "CV_" + str(k))
    if not os.path.exists(path_CV):
        os.mkdir(path_CV)
    path_Network_weights = os.path.join(path_CV, "Network_Weights")
    if not os.path.exists(path_Network_weights):
        os.mkdir(path_Network_weights)
    path_MIPs = os.path.join(path_CV, "MIPs")
    if not os.path.exists(path_MIPs):
        os.mkdir(path_MIPs)
    path_Metrics = os.path.join(path_CV, "Metrics")
    if not os.path.exists(path_Metrics):
        os.mkdir(path_Metrics)

def save_model(model, epoch, optimizer, k, path_Output):
    best_metric_epoch = epoch
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_metric_epoch}
    torch.save(state, os.path.join(path_Output, "CV_" + str(k) + "/Network_Weights/best_model_{}.pth.tar".format(best_metric_epoch)))

def calculate_specificity(confusion_matrix_df, GT_class, pred_class):
    specificity = []
    for cls in GT_class:
        cls_orig = cls.split('_')[0]
        FP = np.sum(np.array(confusion_matrix_df.loc[[i for i in GT_class if i!=cls], [cls_orig + "_Pred"]]))
        TN = np.sum(np.array(confusion_matrix_df.loc[[i for i in GT_class if i!=cls], [i for i in pred_class if i!=cls_orig + "_Pred"]]))
        specificity.append(TN/(TN+FP))
    return specificity

def get_pat_ID_scan_date_t(val_files, withoutTime=False):
    pat_ID = val_files["SUV_MIP"].split("/")[-3]
    scan_date = val_files["SUV_MIP"].split("/")[-2]

    if withoutTime:
        return pat_ID, scan_date
    else:
        time = val_files["time"]
        return pat_ID, scan_date, time
    
def calculate_metrics(pred_prob, GT):
    fpr, tpr, thresholds = metrics.roc_curve(GT, pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (pred_prob >= optimal_threshold).astype(int)
    
    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    # Assuming pred_labels and GT are NumPy arrays or PyTorch tensors
    TP = ((pred_labels == 1) & (GT == 1)).sum()
    TN = ((pred_labels == 0) & (GT == 0)).sum()
    FP = ((pred_labels == 1) & (GT == 0)).sum()
    FN = ((pred_labels == 0) & (GT == 1)).sum()

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0

    auc_score = metrics.auc(fpr, tpr)

    # Compute average precision score
    avg_precision = average_precision_score(GT, pred_prob)

    results = [
        ["True Positives (TP)", TP],
        ["True Negatives (TN)", TN],
        ["False Positives (FP)", FP],
        ["False Negatives (FN)", FN],
        ["Sensitivity", sensitivity],
        ["Precision", precision],
        ["Specificity", specificity],
        ["AUC", auc_score],
        ["Optimal Threshold", optimal_threshold],
        ["Average Precision Score", avg_precision]
    ]
    # Print results in tabular form
    print(tabulate(results, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    return auc_score, avg_precision, pred_labels, optimal_threshold

def huber_loss_function(predictions, targets, delta=5):
    errors = torch.abs(predictions - targets)
    quadratic_term = 0.5 * (errors ** 2)
    linear_term = delta * (errors - 0.5 * delta)
    loss = torch.where(errors < delta, quadratic_term, linear_term)
    return loss.mean()

def plot(metric, path, name="N"):
    epoch = [1 * (i + 1) for i in range(len(metric))]
    plt.plot(epoch, metric)
    plt.xlabel("Number of Epochs")
    plt.ylabel(name)
    plt.savefig(path, dpi=400)

def stratified_split(df_clean, k_fold, k, outcome):
    df_list=[ df_clean[df_clean[outcome]==x].reset_index(drop=True) for x in range(2) ]
    factor_list= [ round(x.shape[0]/k_fold) for x in df_list ]

    if k == (k_fold - 1):
        patients_for_val = []
        for x,f in zip(df_list,factor_list):
            patients_for_val.extend(x[f*k:].patient_ID.tolist())
        df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

    else:
        patients_for_val = []
        for x,f in zip(df_list,factor_list):
            patients_for_val.extend(x[f*k:f*k+f].patient_ID.tolist())
        df_val = df_clean[df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

    df_train = df_clean[~df_clean.patient_ID.isin(patients_for_val)].reset_index(drop=True)

    return df_train, df_val
    
##### Dynamic timeframe generation: training, validation
def create_traindynamictime_df(df, clinical_data_date, outcome, arg_seed=1, number_of_timeframes=6, oversampling_factor=2, n_min = 30, n_max = (30 * 12 * 9) + 30):

    unique_pat_ID = np.unique(df["patient_ID"])
    tf_num = number_of_timeframes #//3

    df_list = [] 
    random.seed(arg_seed)
    for pat_ID in tqdm(unique_pat_ID):
        temp_df_list = []
        temp_df = df[df["patient_ID"] == pat_ID].reset_index(drop=True)

        plus_days_period_list = []

        if np.logical_and(outcome=="Progression_Free_Survival", temp_df.progression_free_days.iloc[0]!=999999): 
            for i in range(1, (tf_num)+1):

                temp_df2 = temp_df.copy()
                baseline_date = temp_df2.scan_date
                plus_days_period = int(random.uniform(n_min, temp_df2["progression_free_days"][0]-1))

                if plus_days_period == 0:
                    plus_days_period = 30

                if plus_days_period in plus_days_period_list:
                    continue
                else:
                    plus_days_period_list.append(plus_days_period)

                limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
                temp_df2["days_period"] = plus_days_period
                temp_df2["limit_date"] = limit_date
                
                temp_df2[outcome] = np.where(plus_days_period >= temp_df2.progression_free_days.iloc[0], 0, 1)

                temp_df_list.append(temp_df2)

            for i in range(1, (tf_num*oversampling_factor)+1):
            
                temp_df2 = temp_df.copy()
                baseline_date = temp_df2.scan_date
                diff_clinicaldatadate_scandate = abs((clinical_data_date - baseline_date.iloc[0])/np.timedelta64(1, 'D'))

                plus_days_period = int(random.uniform(temp_df2["progression_free_days"][0], diff_clinicaldatadate_scandate))
            
                if plus_days_period in plus_days_period_list:
                    continue
                else:
                    plus_days_period_list.append(plus_days_period)

                limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
                temp_df2["days_period"] = plus_days_period
                temp_df2["limit_date"] = limit_date
                
                temp_df2[outcome] = 0

                temp_df_list.append(temp_df2)
        elif temp_df.deceased.iloc[0] == True: 
            for i in range(1, (tf_num)+1):

                temp_df2 = temp_df.copy()
                baseline_date = temp_df2.scan_date
                plus_days_period = int(random.uniform(n_min, temp_df2["survival_days"][0]-1))

                if plus_days_period == 0:
                    plus_days_period = 30

                if plus_days_period in plus_days_period_list:
                    continue
                else:
                    plus_days_period_list.append(plus_days_period)

                limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
                temp_df2["days_period"] = plus_days_period
                temp_df2["limit_date"] = limit_date
                
                temp_df2[outcome] = np.where(limit_date >= temp_df2.deceasedDate, 0, 1)

                temp_df_list.append(temp_df2)

            for i in range(1, (tf_num*oversampling_factor)+1):
            
                temp_df2 = temp_df.copy()
                baseline_date = temp_df2.scan_date
                diff_clinicaldatadate_scandate = abs((clinical_data_date - baseline_date.iloc[0])/np.timedelta64(1, 'D'))

                plus_days_period = int(random.uniform(temp_df2["survival_days"][0], n_max))
            
                if plus_days_period in plus_days_period_list:
                    continue
                else:
                    plus_days_period_list.append(plus_days_period)

                limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
                temp_df2["days_period"] = plus_days_period
                temp_df2["limit_date"] = limit_date
                
                temp_df2[outcome] = 0

                temp_df_list.append(temp_df2)
        else:
            for i in range(1, tf_num+1):
                temp_df2 = temp_df.copy()
                baseline_date = temp_df2.scan_date
                diff_clinicaldatadate_scandate = abs((clinical_data_date - baseline_date.iloc[0])/np.timedelta64(1, 'D'))

                plus_days_period = int(random.uniform(n_min, diff_clinicaldatadate_scandate))

                if plus_days_period in plus_days_period_list:
                    continue
                else:
                    plus_days_period_list.append(plus_days_period)
                                        
                limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
                temp_df2["days_period"] = plus_days_period
                temp_df2["limit_date"] = limit_date
                
                temp_df2[outcome] = 1

                temp_df_list.append(temp_df2)     
            
        
        if len(temp_df_list) == 0:
            print("\nPat_ID: ", pat_ID)
            print("baseline_date: ", baseline_date)

        pat_df = pd.concat(temp_df_list, ignore_index=True).reset_index(drop=True)
        df_list.append(pat_df)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    return final_df

def create_allsurvivaldays_df(df, clinical_data_date, outcome, all_survival_days, n_max = (30 * 12 * 9) + 30):
    
    unique_pat_ID = np.unique(df["patient_ID"])  

    df_list = [] 
    # Seed and retrieve the values

    for pat_ID in tqdm(unique_pat_ID):
        temp_df_list = []

        baseline_date = df[df["patient_ID"] == pat_ID].scan_date.reset_index(drop=True)
        
        diff_clinicaldatadate_scandate = abs((clinical_data_date - baseline_date.iloc[0])/np.timedelta64(1, 'D'))

        plus_days_period_list = []

        for days in all_survival_days:
            temp_df = df[df["patient_ID"] == pat_ID].reset_index(drop=True)
            #random.seed(arg_seed*i)
            #number = int(random.uniform(n_min, n_max))
            
            plus_days_period = days

            if (plus_days_period > diff_clinicaldatadate_scandate) and (temp_df.deceased.iloc[0] != True): # for alive limit the timeframes till clinical data date as we don't have follow up after that
                continue
            
            if plus_days_period in plus_days_period_list:
                continue
            else:
                plus_days_period_list.append(plus_days_period)
                
            limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
            temp_df["days_period"] = plus_days_period
            temp_df["limit_date"] = limit_date
            
            if outcome=="Progression_Free_Survival":
                temp_df[outcome] = np.where(plus_days_period >= temp_df.progression_free_days.iloc[0] and temp_df.progression_free_days.iloc[0]!=999999, 0, 1)
            else:
                temp_df[outcome] = np.where(limit_date >= temp_df.deceasedDate.iloc[0], 0, 1)

            temp_df_list.append(temp_df)
        
        if len(temp_df_list) == 0:
            print("\nPat_ID: ", pat_ID)
            print("baseline_date: ", baseline_date)
            print("all_survival_days: ", sorted(all_survival_days.unique()))

        pat_df = pd.concat(temp_df_list, ignore_index=True).reset_index(drop=True)
        df_list.append(pat_df)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    return final_df

def create_test_allsurvivaldays_df(df, clinical_data_date, outcome, all_survival_days, n_max = (30 * 12 * 9) + 30):
    
    unique_pat_ID = np.unique(df["patient_ID"])  

    df_list = [] 

    for pat_ID in tqdm(unique_pat_ID):
        temp_df_list = []

        baseline_date = df[df["patient_ID"] == pat_ID].scan_date.reset_index(drop=True)
        
        diff_clinicaldatadate_scandate = abs((clinical_data_date - baseline_date.iloc[0])/np.timedelta64(1, 'D'))

        plus_days_period_list = []

        for days in all_survival_days:
            temp_df = df[df["patient_ID"] == pat_ID].reset_index(drop=True)
            
            plus_days_period = days
           
            if plus_days_period in plus_days_period_list:
                continue
            else:
                plus_days_period_list.append(plus_days_period)
                
            limit_date = baseline_date + pd.DateOffset(days=plus_days_period)
            temp_df["days_period"] = plus_days_period
            temp_df["limit_date"] = limit_date

            if outcome=="Progression_Free_Survival":
                temp_df[outcome] = np.where(plus_days_period >= temp_df.progression_free_days.iloc[0] and temp_df.progression_free_days.iloc[0]!=999999, 0, 1)
            else:
                temp_df[outcome] = np.where(limit_date >= temp_df.deceasedDate.iloc[0], 0, 1)

            temp_df_list.append(temp_df)
        
        if len(temp_df_list) == 0:
            print("\nPat_ID: ", pat_ID)
            print("baseline_date: ", baseline_date)
            print("all_survival_days: ", sorted(all_survival_days.unique()))

        pat_df = pd.concat(temp_df_list, ignore_index=True).reset_index(drop=True)
        df_list.append(pat_df)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    return final_df

##### Training and Validation

### Baseline method
def only_train_classification_binary(model, epoch, k, path_Output, train_loader, optimizer, loss_function, device, loss_values, outcome, num_classes=2, additional_loss_weight=0.3):
    m = nn.Softmax(dim=1)
    model.train()
    epoch_loss = 0
    step = 0
    
    pred_prob = []
    GT = []

    for inputs, labels in tqdm(train_loader):
        step += 1

        inputs, labels = inputs.to(device), labels.to(device)

        output_logits = model(inputs)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(output_logits).view(-1)
        
        loss = loss_function(output_logits.view(-1), labels.type(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        #break

    epoch_loss /= step
    loss_values.append(epoch_loss)

    return epoch_loss, loss_values

def only_validation_classification_binary(args, k, epoch, optimizer, loss_function, model, df_val, device, metric_values, path_Output, best_val_loss, loss_values, outcome, channel_flag, num_classes=2):

    pred_prob_os = []

    GT_os = []

    patients_list = []
    scan_date_list = []

    epoch_loss = 0
    step = 0
    pat = 0

    val_files, val_loader = prepare_data_binary(args, df_val, args["batch_size_val"], shuffle=False, label=outcome, channel_flag=channel_flag)

    for inputs, labels in tqdm(val_loader):
        #print("inputs: ", inputs.shape)
        pat_id, scan_date = get_pat_ID_scan_date_t(val_files[pat], withoutTime=True)
        model.eval()
        step += 1

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output_logits= model(inputs)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(output_logits)
        outputs = outputs.data.cpu().numpy()

        loss = focal_loss(output_logits[0], labels.type(torch.float32), loss_function)

        epoch_loss += loss.item()
        pred_prob_os.append(outputs[0][0])
        labels = labels.cpu().numpy()
        GT_os.append(labels[0])

        patients_list.append(pat_id)
        scan_date_list.append(scan_date)
        pat += 1

    epoch_loss /= step
    loss_values.append(epoch_loss)

    #Save the model if validation loss is decreasing
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        save_model(model, epoch, optimizer, k, path_Output)

    print("\n\Overall Survival: ")
    auc, avg_precision, pred_labels_os_list, _ = calculate_metrics(np.array(pred_prob_os).astype(float), np.array(GT_os).astype(int))

    metric_values["AUC"].append(auc)
    metric_values["Avg_Precision"].append(avg_precision)

    df_predictions = pd.DataFrame({'patient_ID': patients_list, 'scan_date': scan_date_list, 
                                    'GT_os': GT_os, 'prediction_os': pred_labels_os_list, 'pred_prob_os': pred_prob_os})

    df_predictions = df_predictions.sort_values(by=['patient_ID'])

    df_predictions.to_csv(os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch) + ".csv"))

    return metric_values, epoch_loss, loss_values, best_val_loss

def testing_classification_binary(args, k, optimizer, loss_function, model, df_val, device, path_Output, outcome, channel_flag, num_classes=2):

    pred_prob_os = []

    GT_os = []

    patients_list = []
    scan_date_list = []

    epoch_loss = 0
    step = 0
    pat = 0

    val_files, val_loader = prepare_data_binary(args, df_val, args["batch_size_val"], shuffle=False, label=outcome, channel_flag=channel_flag)

    for inputs, labels in tqdm(val_loader):
        pat_id, scan_date = get_pat_ID_scan_date_t(val_files[pat], withoutTime=True)
        model.eval()
        step += 1

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output_logits= model(inputs)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(output_logits)
        outputs = outputs.data.cpu().numpy()

        loss = loss_function(output_logits[0], labels.type(torch.float32))

        epoch_loss += loss.item()
        pred_prob_os.append(outputs[0][0])
        labels = labels.cpu().numpy()
        GT_os.append(labels[0])

        patients_list.append(pat_id)
        scan_date_list.append(scan_date)
        pat += 1

    epoch_loss /= step

    print("\n\Overall Survival: ")
    auc, avg_precision, pred_labels_os_list, _ = calculate_metrics(np.array(pred_prob_os).astype(float), np.array(GT_os).astype(int))

    df_predictions = pd.DataFrame({'patient_ID': patients_list, 'scan_date': scan_date_list, 
                                    'GT_os': GT_os, 'prediction_os': pred_labels_os_list, 'pred_prob_os': pred_prob_os})

    df_predictions = df_predictions.sort_values(by=['patient_ID'])

    df_predictions.to_csv(os.path.join(path_Output, "CV_" + str(k), "Testset_Metrics", "best_model.csv"))

    return epoch_loss

### Proposed method
def only_train_classification_binary_withTime(model, epoch, k, path_Output, train_loader, optimizer, loss_function, device, loss_values, outcome, num_classes=2, additional_loss_weight=0.3):
    m = nn.Softmax(dim=1)
    model.train()
    epoch_loss = 0
    step = 0
    
    pred_prob = []
    GT = []

    for inputs, t, labels in tqdm(train_loader):
        step += 1

        t2 = t + 30
        t2 = t2.to(device)

        inputs, t, labels = inputs.to(device), t.to(device), labels.to(device)

        output_logits = model(inputs, t)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(output_logits).view(-1)

        loss1 = focal_loss(output_logits.view(-1), labels.type(torch.float32), loss_function)
        #print("loss1: ", loss1)

        # compute additional loss if sigmoid survival probabilities at time t+30 are higher than prob at time t
        output_logits2 = model(inputs, t2)
        outputs2 = sigmoid(output_logits2).view(-1)

        if any(outputs2 > outputs):
            additional_loss = outputs2 - outputs #prob_diff = torch.abs(probs_pred - targets)
            additional_loss = torch.clamp(additional_loss, min=0)
            additional_loss = torch.mean(additional_loss)
        else:
            additional_loss = 0

        loss = loss1 + additional_loss #additional_loss_weight * additional_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    loss_values.append(epoch_loss)

    return epoch_loss, loss_values

def only_validation_classification_binary_withTime(args, k, epoch, optimizer, loss_function, model, df_val, device, metric_values, path_Output, best_val_loss, loss_values, outcome, channel_flag, num_classes=2):

    unique_pat_ID_scan_date_days_period = np.unique(df_val["unique_pat_ID_scan_date_days_period"])

    pred_prob_os = []

    GT_os = []

    patients_list = []
    scan_date_list = []
    days_period_list = []

    epoch_loss = 0
    step = 0
    pat = 0

    val_files, val_loader = prepare_data_binary_withTime(args, df_val, args["batch_size_val"], shuffle=False, label=outcome, channel_flag=channel_flag)

    for inputs, t, labels in tqdm(val_loader):
        pat_id, scan_date, days_period = get_pat_ID_scan_date_t(val_files[pat])
        model.eval()
        step += 1
        t2 = t + 30
        t2 = t2.to(device)
        inputs, t, labels = inputs.to(device), t.to(device), labels.to(device)
        optimizer.zero_grad()
        output_logits= model(inputs, t)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(output_logits)
        outputs = outputs.data.cpu().numpy()
        loss1 = focal_loss(output_logits[0], labels.type(torch.float32), loss_function)

        output_logits2 = model(inputs, t2)
        outputs2 = sigmoid(output_logits2)
        outputs2 = outputs2.data.cpu().numpy()
        
        if outputs2[0][0] > outputs[0][0]:
            additional_loss = outputs2[0][0] - outputs[0][0]
        else:
            additional_loss = 0

        loss = loss1 + additional_loss #additional_loss_weight * additional_loss
        epoch_loss += loss.item()
        pred_prob_os.append(outputs[0][0])
        labels = labels.cpu().numpy()
        GT_os.append(labels[0])

        patients_list.append(pat_id)
        scan_date_list.append(scan_date)
        days_period_list.append(days_period)
        pat += 1

    epoch_loss /= step
    loss_values.append(epoch_loss)

    #Save the model if validation loss is decreasing
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        save_model(model, epoch, optimizer, k, path_Output)

    print("\n\Overall Survival: ")
    auc, avg_precision, pred_labels_os_list, _ = calculate_metrics(np.array(pred_prob_os).astype(float), np.array(GT_os).astype(int))

    metric_values["AUC"].append(auc)
    metric_values["Avg_Precision"].append(avg_precision)

    df_predictions = pd.DataFrame({'patient_ID': patients_list, 'scan_date': scan_date_list, 'days_period': days_period_list, 
                                    'GT_os': GT_os, 'prediction_os': pred_labels_os_list, 'pred_prob_os': pred_prob_os})

    df_predictions.days_period = df_predictions.days_period.astype(int)

    df_predictions = df_predictions.sort_values(by=['patient_ID', 'days_period'])

    df_predictions.to_csv(os.path.join(path_Output, "CV_" + str(k), "Metrics", "epoch_" + str(epoch) + ".csv"))

    return metric_values, epoch_loss, loss_values, best_val_loss

def testing_classification_binary_withTime(args, k, optimizer, loss_function, model, df_val, device, path_Output, outcome, channel_flag, num_classes=2):

    unique_pat_ID_scan_date_days_period = np.unique(df_val["unique_pat_ID_scan_date_days_period"])

    pred_prob_os = []

    GT_os = []

    patients_list = []
    scan_date_list = []
    days_period_list = []

    epoch_loss = 0
    step = 0
    #metric_values = []
    pat = 0

    val_files, val_loader = prepare_data_binary_withTime(args, df_val, args["batch_size_val"], shuffle=False, label=outcome, channel_flag=channel_flag)

    for inputs, t, labels in tqdm(val_loader):
        #print("inputs: ", inputs.shape)
        pat_id, scan_date, days_period = get_pat_ID_scan_date_t(val_files[pat])
        model.eval()
        step += 1
        t2 = t + 30
        t2 = t2.to(device)
        inputs, t, labels = inputs.to(device), t.to(device), labels.to(device)
        optimizer.zero_grad()
        output_logits= model(inputs, t)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(output_logits)
        outputs = outputs.data.cpu().numpy()
        loss1 = focal_loss(output_logits[0], labels.type(torch.float32), loss_function)

        output_logits2 = model(inputs, t2)
        #print("output_logits2: ", output_logits2.view(-1))
        outputs2 = sigmoid(output_logits2)
        outputs2 = outputs2.data.cpu().numpy()
        
        if outputs2[0][0] > outputs[0][0]:
            additional_loss = outputs2[0][0] - outputs[0][0]
        else:
            additional_loss = 0

        loss = loss1 + additional_loss #additional_loss_weight * additional_loss
        epoch_loss += loss.item()
        pred_prob_os.append(outputs[0][0])
        labels = labels.cpu().numpy()
        GT_os.append(labels[0])

        patients_list.append(pat_id)
        scan_date_list.append(scan_date)
        days_period_list.append(days_period)
        pat += 1

    epoch_loss /= step

    print("\n\Overall Survival: ")
    auc, avg_precision, pred_labels_os_list, _ = calculate_metrics(np.array(pred_prob_os).astype(float), np.array(GT_os).astype(int))

    df_predictions = pd.DataFrame({'patient_ID': patients_list, 'scan_date': scan_date_list, 'days_period': days_period_list, 
                                    'GT_os': GT_os, 'prediction_os': pred_labels_os_list, 'pred_prob_os': pred_prob_os})

    # df_predictions["days_period_norm"] = df_predictions["days_period"]

    # max_val = (30 * 12 * 9) + 30
    # df_predictions["days_period"] = df_predictions["days_period_norm"].apply(lambda x: x*max_val)

    df_predictions.days_period = df_predictions.days_period.astype(int)

    df_predictions = df_predictions.sort_values(by=['patient_ID', 'days_period'])

    df_predictions.to_csv(os.path.join(path_Output, "CV_" + str(k), "Testset_Metrics_withForecast", "best_model.csv"))

    return epoch_loss
