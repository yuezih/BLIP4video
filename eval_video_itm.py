'''
 * This file is used to compute the similarity between given videos and generated captions (for re-ranking). 
 * By Zihao Yue
'''

import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_itm import blip_itm
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

import pdb

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    

    print_freq = 10

    sim_all = []
    for i, (video, caption, video_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video = video.to(device)
        sim = model(video, caption, match_head='itc')
        for i in range(sim.shape[0]):
            sim_all.append({
                'video_id': video_id[i].item(),
                'sim': sim[i][i].item()
            })
    return sim_all

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    val_dataset = create_dataset('itm', config)  

    samplers = [None]
    
    val_loader = create_loader([val_dataset], samplers, 
                                batch_size=[config['batch_size_test']],
                                num_workers=[4],
                                is_trains=[False], 
                                collate_fns=[None])[0]

    #### Model #### 
    print("Creating model")
    model = blip_itm(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    print("Start evaluating.")
    start_time = time.time()     
            
    val_result = evaluation(model_without_ddp, val_loader, device, config)
    with open('/data2/yzh/BLIP_video/output/video_itm/sim_result.json', 'w') as f:
        json.dump(val_result, f)

    dist.barrier()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)