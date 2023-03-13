<div>
  <h2 align="center">
    <img src="https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/blip.png" width="40" />
      BLIP4video: BLIP Captioner's Video Solution
  </h2>
</div>

<p align="center">
    <a >
       <img alt="Issues" src="https://img.shields.io/github/issues/yuezih/BLIP4video?color=blueviolet" />
  	</a>
    <a >
       <img alt="Forks" src="https://img.shields.io/github/forks/yuezih/BLIP4video?color=orange" />
  	</a>
    <a >
       <img alt="Stars" src="https://img.shields.io/github/stars/yuezih/BLIP4video?color=ff69b4" />
  	</a>
<a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
    <br />
</p>

<div align="center">
      <img src="https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/blip4video.png" width=500>
</div>

This is the PyTorch code of BLIP4video, a modified version of [BLIP](https://github.com/salesforce/BLIP) for the Video-to-Text Description (VTT) task at TRECVID 2022. Our submission ranks 1st in all official evaluation metrics including BLEU, METEOR, CIDER, SPICE, and STS, and achieves the best submission score of 60.2 on CIDEr, 67.2\% higher than last year’s best result.  

<div align="center">
      <img src="https://yuezih-bucket.oss-cn-beijing.aliyuncs.com/leaderboard.png" width="500">
</div>

### Catalog:
- [x] BLIP captioner's video solution
- [x] Self-critical reinforcement learning for video captioning (VinVL implementation)
- [x] Text-video retrieval and matching for caption candidates scoring and re-ranking


### Video-Text Captioning:
1. Set data root in configs/*.yaml accordingly.
2. To train the finetuned BLIP4video model for the video captioning task, run:  
```
bash scripts/train_video_caption.sh
```


### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@article{yuerucaim3,
  title={RUCAIM3-Tencent at TRECVID 2022: Video to Text Description},
  author={Yue, Zihao and Liu, Yuqi and Zhang, Liang and Yao, Linli and Jin, Qin}
}</pre>

### Acknowledgement
The implementation of BLIP relies on resources from [BLIP](https://github.com/salesforce/BLIP) and [Oscar](https://github.com/microsoft/Oscar). We thank the original authors for their open-sourcing.  
