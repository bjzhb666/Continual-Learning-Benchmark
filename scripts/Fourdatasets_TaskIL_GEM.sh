GPUID=4
OUTDIR=outputs/Fourdatasets_TaskIL
REPEAT=1
mkdir -p $OUTDIR
python  -u iBatchLearn5data.py --gpuid $GPUID --repeat $REPEAT \
--optimizer Adam  --dataset fourdatasets  \
--force_out_dim 0  \
--schedule 80 120 160 --batch_size 32 \
--model_name ResNet18torch --model_type resnet \
--agent_type customization  --agent_name GEM_4000        \
--mydataroot /home/nibolin/zhaohongbo/Continual-Learning-Benchmark/data/data/ \
--lr 0.01 --reg_coef 0.5     | tee ${OUTDIR}/GEM4000_reg_coef=0.5.log