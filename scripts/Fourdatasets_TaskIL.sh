GPUID=1
OUTDIR=outputs/Fourdatasets_TaskIL
REPEAT=1
mkdir -p $OUTDIR
python -u iBatchLearn5data.py --gpuid $GPUID --repeat $REPEAT \
--optimizer Adam  --dataset fourdatasets  \
--force_out_dim 0  \
--schedule 80 120 160 --batch_size 64 \
--model_name ResNet18 --model_type resnet \
--agent_type customization  --agent_name EWC        \
--lr 0.001 --reg_coef 1       | tee ${OUTDIR}/ EWC_reg_coef=1.log