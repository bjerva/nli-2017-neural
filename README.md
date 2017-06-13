# nli-2017-neural

python3 train.py -g0 -o results/ --dataset data/nli/nli-shared-task-2017/data/ --out_size 64 --hidden_size 128 --maxlen 4096 --batchsize 32 --epoch 1000
 --dropout 0.6 --bn --activation --subset --use_bow

This yields about 80% on dev after 100 epochs (on the subsets) 
