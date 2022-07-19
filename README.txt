Env
conda install --yes --file requirements.txt

Pretrain_Model
https://drive.google.com/file/d/1Z3bQibe40XeoXT4QOFWNcDpE4C6fBCxO/view?usp=sharing
put it in the root path of this project

Train with Adaptive Threshold
bash train_adaptive_threshold.sh

Train with Optimal Threshold
bash train_optimal_threshold.sh

You should change the 'data_dir' and 'gt_dir' in the script before training
