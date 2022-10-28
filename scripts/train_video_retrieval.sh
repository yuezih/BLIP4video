CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.run \
--nproc_per_node=4 \
--master_port 30003 \
train_video_retrieval.py \
--config ./configs/retrieval_video.yaml \
--output_dir output/retrieval_video