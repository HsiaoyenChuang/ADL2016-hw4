machine translate的使用：
训练模型：
python translate.py --data_dir data/  --train_dir model_mt/

跑测试数据：
python translate.py --data_dir data/ --decode True  --train_dir model_mt/

nature language generation的使用：
训练模型：
python nlg.py --data_dir data/ --train_dir model_nlg/

跑测试数据
python nlg.py --data_dir data/ --decode True --train_dir model_nlg/