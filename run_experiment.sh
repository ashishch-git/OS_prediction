#!/bin/bash

### Sequential baseline (Cross Validation)
/home/ashish/anaconda3/envs/dlfia/bin/python classification_binary_sequencial_Train_Validation_baseline.py --year 2 --exp_seed 0 --model_name "ResNet-50" --channel_flag "SUV_CT_T_12" --fold 0 --max_epochs 50 --augmentation True

wait

### baseline (Testing)
/home/ashish/anaconda3/envs/dlfia/bin/python classification_binary_Test_baseline.py --year 2 --exp_seed 0 --model_name "ResNet-50" --channel_flag "SUV_CT_T_12" --fold 0

wait

### Sequential proposed 1 (Cross Validation)
/home/ashish/anaconda3/envs/dlfia/bin/python classification_binary_sequencial_Train_Validation_proposed_1.py --exp_seed 0 --model_name "ResNet-50" --channel_flag "SUV_CT_T_12" --fold 0 --max_epochs 50 --augmentation True

wait

###  proposed 1 (Testing)
/home/ashish/anaconda3/envs/dlfia/bin/python classification_binary_Test_proposed_1.py --exp_seed 0 --model_name "ResNet-50" --channel_flag "SUV_CT_T_12" --fold 0
