#!/bin/bash
data_dir=data/
#train_dir=model/
decode=True

python nlg.py --data_dir $data_dir \
      --decode $decode \
      #--train_dir   $model_dir \
