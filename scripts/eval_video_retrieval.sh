CUDA_VISIBLE_DEVICES="6,7" \
python -m torch.distributed.run \
--nproc_per_node=2 train_video_retrieval.py \
--config ./configs/retrieval_video.yaml \
--output_dir output/retrieval_video \
--evaluate