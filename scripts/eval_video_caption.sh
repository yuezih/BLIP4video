CUDA_VISIBLE_DEVICES="4,5,6,7" \
python -m torch.distributed.run \
--nproc_per_node=4 \
--master_port 30002 \
train_video_caption.py \
--evaluate \
--config ./configs/eval_vtt.yaml \
--output_dir output/Caption_vtt_eval