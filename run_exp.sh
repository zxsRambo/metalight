#!/usr/bin/bash

memo="meta_train"
### **************** meta_train ********************
## MetaLight
python meta_train.py --memo ${memo} --algorithm MetaLight 
echo "metalight: meta_train complete"

## MAML 
#python meta_train.py --memo ${memo} --algorithm FRAPPlus 
#echo "maml: meta_train complete"

## Pretrained
#python meta_test.py --memo ${memo} --algorithm FRAPPlus --multi_episodes --run_round 25 --num_process 2 
#echo "Pre-trained: meta_train complete"

### Put the model trained by MetaLight, MAML and Pretrained into 'model/initial/common/' and Perform meta-test
### **************** meta_test ********************
#memo="meta_test"
#model_type="metalight" # maml or pretrained or random
#traffic_group="train_all" # test_all, valid_all or city_all
#python meta_test.py --memo ${memo} --algorithm FRAPPlus --num_process 2 --pre_train --pre_train_model_name ${model_type}  --run_round 1 --num_process 2 --update_start 0 --test_period 1 --traffic_group ${traffic_group} 
#python summary.py --memo ${memo} --type meta_test 

### **************** SOTL ********************
#python run_sotl.py --memo ${memo}
#python summary.py --memo ${memo} --type sotl