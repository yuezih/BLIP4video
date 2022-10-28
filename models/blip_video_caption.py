'''
File: blip_video_caption.py
Author: Yuqi Liu, Zihao Yue
'''
from models.med import BertConfig, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
from models.video_swin_transformer import create_video_swin_transformer
from models.vit import Block

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from torch.autograd import Variable

import faiss
import pickle

from losses.caption_evaluate import ScstRewardCriterion

import pdb

class BLIP_Video_Caption(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.video_swin, video_swin_width = create_video_swin_transformer('base')
        # self.demension_decay = nn.Sequential(nn.Linear(video_swin_width, vision_width), nn.ReLU(True))
        # self.temporal_transformer = TemporalTransformer(vision_width)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)
        self.scst_criterion = ScstRewardCriterion()
        # self.mem_bank = pickle.load(open('/data2/yzh/BLIP_video/memory_bank/mem_bank.pkl','rb'))
        # self.faiss_index = faiss.read_index('/data2/yzh/BLIP_video/memory_bank/trainset.index')
        # # gpu acceleration
        # self.res = faiss.StandardGpuResources()
        # self.faiss_index = faiss.index_cpu_to_gpu(self.res, 0, self.faiss_index)
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, video, caption, training_mode='xe', config=None):
        # for vit
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H) # shape is (B*N, C, W, H)
        video_embeds = self.visual_encoder(video) # shape is (B*N, patch_len, hidden_dim)
        video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]).view(B,-1,video_embeds.shape[-1])
        # video_embeds = self.temporal_transformer(video_embeds)
        
        # # for video_swin
        # B,N,C,W,H = video.size()
        # video = video.permute(0,2,1,3,4) # shape is (B, C, N, W, H)
        # video_embeds = self.video_swin(video)  # output shape is [B, hid_dim, N//2, h, w]
        # video_embeds = video_embeds.reshape(video_embeds.shape[:2]+(-1,))
        # video_embeds = video_embeds.permute(0,2,1)  # output shape is [B, patches, hid_dim]
        # video_embeds = self.demension_decay(video_embeds)

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video.device)
        
        if training_mode == 'xe':
            text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(video.device) 
            text.input_ids[:,0] = self.tokenizer.bos_token_id
            
            decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
            decoder_targets[:,:self.prompt_length] = -100
        
            decoder_output = self.text_decoder(text.input_ids, 
                                            attention_mask = text.attention_mask, 
                                            encoder_hidden_states = video_embeds,
                                            encoder_attention_mask = video_atts,                  
                                            labels = decoder_targets,
                                            return_dict = True,   
                                            )
            loss_lm = decoder_output.loss
        
        elif training_mode == 'rl':
            model_kwargs = {"encoder_hidden_states": video_embeds, "encoder_attention_mask":video_atts}
            prompt = [self.prompt] * B
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(video.device) 
            input_ids[:,0] = self.tokenizer.bos_token_id
            input_ids = input_ids[:, :-1]

            # greedy
            with torch.no_grad():
                outputs_max = self.text_decoder.generate(input_ids=input_ids, max_length=config['max_length'], min_length=config['min_length'], num_beams=1, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.tokenizer.pad_token_id, repetition_penalty=1.0, output_scores=True, return_dict_in_generate=True, **model_kwargs)
                sents_max = outputs_max.sequences[:, 1:]
                captions_max = self.caption_decode(sents_max)
            # sample
            sample_num = 4
            video_embeds = video_embeds.repeat_interleave(sample_num, dim=0)
            video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video.device)
            model_kwargs = {"encoder_hidden_states": video_embeds, "encoder_attention_mask":video_atts}
            outputs_sam = self.text_decoder.generate(input_ids=input_ids, max_length=config['max_length'], min_length=config['min_length'], do_sample=True, top_p=0.9, num_return_sequences=sample_num, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.tokenizer.pad_token_id, repetition_penalty=1.0, output_scores=True, return_dict_in_generate=True, **model_kwargs)
            sents_sam = outputs_sam.sequences[:, 1:]
            captions_sam = self.caption_decode(sents_sam)
            
            # scores_sam = outputs_sam.scores[:, 1:sents_sam.shape[1]+1]
            # scores_sam = F.log_softmax(scores_sam, dim=-1)
            scores_sam = torch.stack(outputs_sam.scores, 1)
            scores_sam = F.log_softmax(scores_sam, dim=-1)
            scores_sam = torch.gather(scores_sam, 2, sents_sam.unsqueeze(-1))
            scores_sam = scores_sam.squeeze(-1)

            gt_res = [[c] for c in caption]
            mask = sents_sam > 0
            logP_sam_mask = scores_sam * mask
            logP_sam_mask = torch.where(torch.isnan(logP_sam_mask), torch.full_like(logP_sam_mask, 0), logP_sam_mask)
            sample_logprobs = logP_sam_mask.sum(-1)
            sample_cnt = mask.sum(-1)
            sample_logprobs = sample_logprobs / sample_cnt
            # logP_avg = Variable(torch.zeros(logP_sam.shape[0]).cuda())
            # for i in range(logP_sam.shape[0]):
            #     logP_avg[i] = torch.mean(logP_sam[i][mask[i]])

            loss_lm = self.scst_criterion(gt_res, captions_max, captions_sam, sample_logprobs)
            
        return loss_lm
        
    def generate(self, video, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        # for vit
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H) # shape is (B*N, C, W, H)
        video_embeds = self.visual_encoder(video) # shape is (B*N, patch_len, hidden_dim)
        video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]).view(B,-1,video_embeds.shape[-1]) # shape is (B, N*patch_len, hidden_dim)
        # video_embeds = self.temporal_transformer(video_embeds)

        # # for video_swin
        # B,N,C,W,H = video.size()
        # video = video.permute(0,2,1,3,4) # shape is (B, C, N, W, H)
        # video_embeds = self.video_swin(video)  # output shape is [B, hid_dim, N//2, h, w]
        # video_embeds = video_embeds.reshape(video_embeds.shape[:2]+(-1,))
        # video_embeds = video_embeds.permute(0,2,1)  # output shape is [B, patches, hid_dim]
        # video_embeds = self.demension_decay(video_embeds)

        if not sample:
            video_embeds = video_embeds.repeat_interleave(num_beams, dim=0)
            
        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video.device)
        model_kwargs = {"encoder_hidden_states": video_embeds, "encoder_attention_mask":video_atts}
        
        prompt = [self.prompt] * B
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(video.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        # maic = False
        # if maic:
        #     # decoder generation
        #     outputs = self.text_decoder.generate(input_ids=input_ids, max_length=max_length, min_length=min_length, num_beams=num_beams, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.tokenizer.pad_token_id, repetition_penalty=repetition_penalty, output_hidden_states=True, output_scores=True, return_dict_in_generate=True, **model_kwargs)
        #     sequences = outputs.sequences[:, 1:]
        #     raw_distribution = torch.softmax(torch.stack(outputs.scores, 1), dim=-1) # (B, seq_len, vocab_size)
        #     pdb.set_trace()
        #     # retrieval results
        #     dec_query = outputs.hidden_states[-1][:,:-1] # (B, 1+seq_len+pad, hidden_dim) torch.Size([10, 32, 768])
        #     enc_query = video_embeds.unsqueeze(1).view(B,-1,197,768)[:,:,0]
        #     enc_query = enc_query[:,0] # (B, hidden_dim)
        #     for sent_id in range(dec_query.shape[0]):
        #         for each_dq in dec_query[sent_id]:
        #             query = torch.cat([enc_query[sent_id], each_dq], dim=-1).unsqueeze(0)
        #             rtr_res = self.faiss_index.search(query, 512)
        #             rtr_dis = torch.tensor(rtr_res[0])
        #             rtr_idx = torch.tensor(rtr_res[1])
        #             # weight = exp(-dis/T) * lambda
        #             rtr_weight = torch.exp(-rtr_dis/1000) * 0.5 # 约0.17


        # else:
        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                max_length=max_length,
                                                min_length=min_length,
                                                do_sample=True,
                                                top_p=top_p,
                                                num_return_sequences=1,
                                                eos_token_id=self.tokenizer.sep_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id, 
                                                repetition_penalty=1.1,                                            
                                                **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                max_length=max_length,
                                                min_length=min_length,
                                                num_beams=num_beams,
                                                num_return_sequences=5,
                                                eos_token_id=self.tokenizer.sep_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id,     
                                                repetition_penalty=repetition_penalty,
                                                **model_kwargs)      
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions


    def caption_decode(self, sents):
        captions = []
        for sent in sents:
            caption = self.tokenizer.decode(sent, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions


    def generate_key(self, video, caption):
        # for vit
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H) # shape is (B*N, C, W, H)
        video_embeds = self.visual_encoder(video) # shape is (B*N, patch_len, hidden_dim)
        video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]).view(B,-1,video_embeds.shape[-1]) # shape is (B, N*patch_len, hidden_dim)

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(video.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        ''' Version-1: <CLS> token
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = video_embeds,
                                           encoder_attention_mask = video_atts,                  
                                           labels = decoder_targets,
                                           output_hidden_states = True,
                                           return_dict = True,
                                          )
        
        dec_output = decoder_output.hidden_states[-1][:,:-1] # (B, 1+seq_len+pad, hidden_dim) torch.Size([10, 32, 768])
        enc_output = video_embeds.unsqueeze(1).view(B,-1,197,768)[:,:,0]
        enc_output = enc_output[:,0].unsqueeze(1).repeat(1,dec_output.shape[1],1) # torch.Size([10, 32, 768])
        # dec_output = dec_output.unsqueeze(2).repeat(1,1,64,1)
        '''

        # ''' Version-2: attention
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = video_embeds,
                                           encoder_attention_mask = video_atts,                  
                                           labels = decoder_targets,
                                           output_hidden_states = True,
                                           output_attentions = True,
                                           return_dict = True,
                                          )

        dec_output = decoder_output.hidden_states[-1][:,:-1]
        vis_attn = decoder_output.cross_attentions[-1].mean(dim=1)[:,:-1] # (B, seq_len, patch_len)
        patch_select = vis_attn.argmax(dim=-1) # (B, seq_len)
        enc_output_all = video_embeds.unsqueeze(1).repeat(1,dec_output.shape[1],1,1) # (B, seq_len, patch_len, hidden_dim)
        patch_indices = patch_select.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,video_embeds.shape[-1])
        # enc_output[i][j][0][k] = enc_output_all[i][j][patch_indices[i][j][0][k]][k]
        enc_output = enc_output_all.gather(dim=-2, index=patch_indices).squeeze(-2) # (B, seq_len, hidden_dim)

        key = torch.cat([enc_output, dec_output], dim=2) # torch.Size([10, 32, 1536])
        key = key.view(-1, key.shape[-1]) # (B*seq_len, key_len)
        value = decoder_targets[:,1:].flatten() # (B*seq_len)

        return key, value


class TemporalTransformer(nn.Module):
    def __init__(self, vision_width=384):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=vision_width, num_heads=12
            )
            for i in range(2)])
    def forward(self, x):
        for i,blk in enumerate(self.blocks):
            x = blk(x)
        return x

def blip_video_caption(pretrained='',**kwargs):
    model = BLIP_Video_Caption(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print(msg.missing_keys)
        assert(len(msg.missing_keys)==0)
    return model
