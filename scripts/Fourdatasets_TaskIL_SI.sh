GPUID=4
OUTDIR=outputs/Fourdatasets_TaskIL
REPEAT=1
mkdir -p $OUTDIR
python  -u iBatchLearn5data.py --gpuid $GPUID --repeat $REPEAT \
--optimizer Adam  --dataset fourdatasets  \
--force_out_dim 0  \
--schedule 80 120 160 --batch_size 32 \
--model_name ResNet18torch --model_type resnet \
--agent_type regularization  --agent_name SI        \
--lr 0.001 --reg_coef 10     | tee ${OUTDIR}/SI_reg_coef=10.log