# PanSplat

### PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting
Cheng Zhang, Haofei Xu, Qianyi Wu, Camilo Cruz Gambardella, Dinh Phung, Jianfei Cai

### [Project Page](https://chengzhag.github.io/publication/pansplat) | [Paper](http://arxiv.org/abs/2412.12096)

![teaser](images/teaser.png)

## Introduction

This repo contains training, testing, evaluation code of our arXiv 2024 paper.

## Installation

We use Anaconda to manage the environment. You can create the environment by running the following command:

```bash
conda create -n pansplat python=3.10
conda activate pansplat
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
pip3 install -U xformers==0.0.27.post2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

We use wandb to log and visualize the training process. You can create an account then login to wandb by running the following command:
```bash
wandb login
```

## Demo

You can download the pretrained checkpoints [last.ckpt](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/EUSd23tEyjpIg-A6YMdrV-gBMSHG9hLk5zYC_Aq80csDig?e=0gMnFr) (trained on the Matterport3D dataset at 512 × 1024 resolution) and put it in the `logs/nvpl49ge/checkpoints` folder. Then run the following command to test the model:
    
```bash
python -m src.paper.demo +experiment=pansplat-512 ++model.weights_path=logs/nvpl49ge/checkpoints/last.ckpt mode=predict
```

The code will use the sample images in the `datasets/pano_grf` folder:

<img src="datasets/pano_grf/png_render_test_1024x512_seq_len_3_m3d_dist_0.5/00000007/00/rgb.png" alt="demo_input_image1" width="49%"> <img src="datasets/pano_grf/png_render_test_1024x512_seq_len_3_m3d_dist_0.5/00000007/02/rgb.png" alt="demo_input_image2" width="49%">

The output will be saved in the folder with the format `outputs/2025-01-13/16-56-04`:
![demo_output_video](images/demo_output_video.gif)

Additionally, we provide a fine-tuned checkpoint [last.ckpt](https://monashuni-my.sharepoint.com/:u:/g/personal/cheng_zhang_monash_edu/Ee1xYAdyL3xOoGZdyMi4fPMBnq5n-XXQmGZvSrirAhrjGA?e=mU2pAR) (fine-tuned on the Matterport3D dataset at 2048 × 4096 resolution) for 4K panorama synthesis. You can put it in the `logs/hxlad5nq/checkpoints` folder and run the following command to test the model:

```bash
python -m src.paper.demo +experiment=pansplat-2048 ++model.weights_path=logs/hxlad5nq/checkpoints/last.ckpt mode=predict
```

This requires a GPU with at least 24GB of memory, e.g., NVIDIA RTX 3090.

## Data Preparation

### PanoGRF Data

We use the data preparation code from the [PanoGRF](https://github.com/thucz/PanoGRF) repo to render the Matterport3D dataset and generate the Replica and Residential datasets. Please download `pano_grf_lr.tar` from [link](https://monashuni-my.sharepoint.com/:f:/g/personal/cheng_zhang_monash_edu/En3qfWTyLaNGvQnqPcA8xLYBzr3kOReqAbbgINsWXkwaMA?e=DH1M5x) and unzip it to the `datasets` folder.
We also rendered a smaller Matterport3D dataset with higher resolution for fine-tuning. If you plan to fine-tune the model at higher resolution, please download `pano_grf_hr.tar` and unzip it to the `datasets` folder.

### 360Loc Data

We use the [360Loc](https://github.com/HuajianUP/360Loc?tab=readme-ov-file) dataset for fine-tuning to real-world data. Please download the data from the [official link](https://hkustconnect-my.sharepoint.com/personal/cliudg_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcliudg%5Fconnect%5Fust%5Fhk%2FDocuments%2F360Loc%5Frelease&ga=1) and unzip the separate parts to the `datasets/360Loc` folder.

### Our Video Data

We provide two sample videos for testing cross-dataset generalization. Please download `insta360.tar` from [link](https://monashuni-my.sharepoint.com/:f:/g/personal/cheng_zhang_monash_edu/En3qfWTyLaNGvQnqPcA8xLYBzr3kOReqAbbgINsWXkwaMA?e=DH1M5x) and unzip it to the `datasets` folder.

<details>
<summary>Use your own video</summary>
Coming soon...
</details>
<br>

## Training and Testing

### Pretrained Models

We use part of the pretrained UniMatch weights from MVSplat and the pretrained panoramic monocular depth estimation model from PanoGRF. Please download the [weights](https://monashuni-my.sharepoint.com/:f:/g/personal/cheng_zhang_monash_edu/EpfWBCHaSzFLpEAJ3EHYtK0BK80YmLxN2yAKu6hfwWatmA?e=qZK2Xc) and put them in the `checkpoints` folder.

### Train on Matterport3D

We train the model on the Matterport3D dataset starting from a low resolution and fine-tune it at higher resolutions. You can run the following command to train the model at 256 × 512 resolution:

```bash
python -m src.main +experiment=pansplat-256 mode=train
```

**Hint:** The training takes about 1 day on a single NVIDIA A100 GPU. Experiments are logged and visualized to wandb under the pansplat project. You'll get a WANDB_RUN_ID (e.g., ek6ab466) after running the command. Or you can find it in the wandb dashboard. At the end of the training, the model will be tested and the evaluation results will be logged to wandb as table. The checkpoints are saved in the logs/<WANDB_RUN_ID>/checkpoints folder. Same for the following experiments.

Please then replace the `model.weights_path` parameter of `config/pansplat-512.yaml` with the path to the last checkpoint of the 256 × 512 resolution training and run the following command to fine-tune the model at 512 × 1024 resolution:

```bash
python -m src.main +experiment=pansplat-512 mode=train
```

<details>
<summary>If you want to fine-tune on high resolution Matterport3D data...</summary>

Similarly, update the `model.weights_path` settings in `config/pansplat-1024.yaml` and fine-tune the model at 1024 × 2048 resolution:

```bash
python -m src.main +experiment=pansplat-1024 mode=train
```

Finally, update the `model.weights_path` settings in `config/pansplat-2048.yaml` and fine-tune the model at 2048 × 4096 resolution:

```bash
python -m src.main +experiment=pansplat-2048 mode=train
```

</details>
<br>


### Fine-tune on 360Loc

We fine-tune the model on the 360Loc dataset from the weights trained on the Matterport3D dataset at 512 × 1024 resolution. Please update the `model.weights_path` parameter of `config/pansplat-512-360loc.yaml` then run the following command:

```bash
python -m src.main +experiment=pansplat-512-360loc mode=train
```

We then gradually increase the resolution to 1024 × 2048 and 2048 × 4096 and fine-tune from the lower resolution weights:

```bash
python -m src.main +experiment=pansplat-1024-360loc mode=train
python -m src.main +experiment=pansplat-2048-360loc mode=train
```

Remember to update the `model.weights_path` parameter in the corresponding config files before running the commands.

## Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{zhang2024pansplat4kpanoramasynthesis,
      title={PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting}, 
      author={Cheng Zhang and Haofei Xu and Qianyi Wu and Camilo Cruz Gambardella and Dinh Phung and Jianfei Cai},
      year={2024},
      eprint={2412.12096},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.12096}, 
}
```
