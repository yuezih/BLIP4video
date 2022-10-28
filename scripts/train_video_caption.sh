CUDA_VISIBLE_DEVICES="0,1,2,3" \
TORCH_DISTRIBUTED_DEBUG=INFO \
python -m torch.distributed.run \
--nproc_per_node=4 \
--master_port 30001 \
train_video_caption.py \
--config ./configs/caption_vtt.yaml \
--output_dir output/Caption_vtt
