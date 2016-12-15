#!/bin/bash
data_dir=data/
#train_dir=model_tmp/
decode=True

python translate.py --data_dir $data_dir \
      --decode $decode \
      #--train_dir   $model_dir \
