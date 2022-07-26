cd best_p3_reprod
CUDA_VISIBLE_DEVICES=1 python ./gru_torch_ens/train.py
CUDA_VISIBLE_DEVICES=1 python ./gru_torch_ens/train.py continuous_training
CUDA_VISIBLE_DEVICES=1 python ./lgb_tune/train.py

cd ..
rm -rf submit.zip
zip -q -r submit.zip best_p3_reprod
#CUDA_VISIBLE_DEVICES=5 python evaluation.py