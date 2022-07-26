rm -rf submit.zip
zip -q -r submit.zip best_p3_reprod
CUDA_VISIBLE_DEVICES=1 python evaluation.py