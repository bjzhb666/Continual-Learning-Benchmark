GPUID=1
OUTDIR=outputs/officehome_domainIL
REPEAT=1
mkdir -p $OUTDIR
python -u iBatchLearn5data.py --gpuid $GPUID --repeat $REPEAT \
--optimizer Adam  --dataset officehome  \
--force_out_dim 65  \
--schedule 80 120 160 --batch_size 64 \
--model_name ResNet18 --model_type resnet \
--agent_type regularization  --agent_name SI        \
--mydataroot /home/nibolin/zhaohongbo/Continual-Learning-Benchmark/data/office-home-dataset/OfficeHomeDataset_10072016/ \
--lr 0.001 --reg_coef 10       | tee ${OUTDIR}/SI_reg_coef=10.log