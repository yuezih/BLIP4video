CUDA_VISIBLE_DEVICES="4,5,6,7" \
python -m torch.distributed.run \
--nproc_per_node=4 \
--master_port 30000 \
eval_video_itm.py \
--config ./configs/eval_itm.yaml \
--output_dir output/video_itm \
--evaluate