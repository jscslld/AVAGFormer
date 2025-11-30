SESSION=$1
CONFIG=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=8 \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/$SESSION/train.py $CONFIG
