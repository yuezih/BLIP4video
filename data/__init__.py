import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.video_caption_dataset import video_caption_train, video_caption_eval
from data.video_retrieval_dataset import video_retrieval_train, video_retrieval_eval
from data.video_itm_dataset import video_itm_eval
from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='video_caption': 
        train_dataset = video_caption_train(transform_test, config['video_root'], config['ann_root'], prompt=config['prompt'], max_img_size=config['image_size'], num_frm=config['num_frm'])
        val_dataset = video_caption_eval(transform_test, config['video_root'], config['ann_root'], 'val', max_img_size=config['image_size'], num_frm=config['num_frm'])
        test_dataset = video_caption_eval(transform_test, config['video_root'], config['ann_root'], 'test', max_img_size=config['image_size'], num_frm=config['num_frm'])   
        return train_dataset, val_dataset, test_dataset
    
    elif dataset == 'retrieval_trecvid':
        train_dataset = video_retrieval_train(transform_test, config['video_root'], config['ann_root'], max_img_size=config['image_size'], num_frm=config['num_frm'])
        val_dataset = video_retrieval_eval(transform_test, config['video_root'], config['ann_root'], 'val', max_img_size=config['image_size'], num_frm=config['num_frm'])
        test_dataset = video_retrieval_eval(transform_test, config['video_root'], config['ann_root'], 'test', max_img_size=config['image_size'], num_frm=config['num_frm'])   
        return train_dataset, val_dataset, test_dataset
    
    elif dataset == 'itm':
        val_dataset = video_itm_eval(transform_test, config['video_root'], config['ann_root'], max_img_size=config['image_size'], num_frm=config['num_frm'])
        return val_dataset

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

