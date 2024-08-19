CUDA_VISIBLE_DEVICES=3 python fl_pretrain.py \
--dataset="mnist" --partition="iid"  --beta=1.0 --seed=42 --num_users=3 \
--model="lenet" \
--local_lr=0.01 --local_ep=1 \
--sigma 0.0 \
--batch_size 128
