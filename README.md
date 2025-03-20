# PointLiBR
Official PyTorch implementation for the following papers:
- **A Large-Scale Network Construction and Lightweighting Method for Point Cloud Semantic Segmentation**  
IEEE Transactions on Image Processing (**TIP**) 2024

- **Subspace Prototype Guidance for Mitigating Class Imbalance in Point Cloud Semantic Segmentation**  
European Conference on Computer Vision (**ECCV**) 2024

_by [Jiawei Han](https://github.com/Javion11)_


## Highlights
-  :boom: 2024/03: **LPFP&PCLN** accepted by TIP'2024. **LPFP** can construct a high-performance large-scale point cloud semantic segmentation model, and **PCLN** can effectively compress the model scale.
-  :boom: 2024/07: **SPG** accepted by ECCV'2024. **SPG** can mitigate class imbalance in point cloud semantic segmentation through separate subspace prototypes.
-  :boom: 2024/07: Code released! This code base is a reference implementation of the main ideas of the papers.
-  :boom: 2025/03: **InvSapceNet** accepted by TPAMI'2025. **InvSapceNet** generates an inverse feature space, which can guide the training of the semantic segmentation network and mitigate the cognitive bias caused by unbalanced data.


## Overview
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)



## Installation
### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.
```bash
source ./install.sh
```

## Data Preparation
Here is the preprocessing method of S3DIS. For more datasets of point cloud semantic segmentation, you can refer to [Pointcept](https://github.com/Pointcept/Pointcept.git).
- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
- Run preprocessing code for S3DIS as follows:
```
python preprocess/s3dis/collect_indoor3d_data.py  
python preprocess/s3dis/indoor3d_util.py
```
- Organize the dataset as follows:
```
data
 |--- S3DIS
      |--- raw
            |--- Area_6_pantry_1.npy
            |--- ...
      |--- processed
            |--- s3dis_val_area5_0.040.pkl 
```

## Quick Start
- **Ease of Use**: _Build_ model, optimizer, scheduler, loss function, and data loader _easily from cfg_. Train and validate different models on various tasks by simply changing the `cfg/*/*.yaml` file. 
- **Main Function**: All main functions are placed in the _main_ folder for easy debugging and training.


## Model Zoo
The **pretrained models** in column `Released`.

### S3DIS (Area 5) Semantic Segmentation
| Model | Additional Data | Num GPUs | OA | mAcc | mIoU | Config | Released |
| :---: |:---------------:| :---: | :---: | :---: | :---: | :---: | :---: |
| **LPFP**(PointNet++*) | &cross; | 2 | 89.9% | 74.6% | 67.7% | [link](cfgs/lpfp_pcln/lpfp_pointnet++_t.yaml) | [link](https://drive.google.com/file/d/1spp88opaDF0t4VMmRSXtgQGOMLBEBPzF/view?usp=drive_link) |
| **PCLN**(**LPFP**(PointNet++*_0.5x)) | &cross; | 2 | 89.6% | 73.1% | 66.4% | [link](cfgs/lpfp_pcln/pcln_pointnet++.yaml) | [link](https://drive.google.com/file/d/1YM9vqeFPXinHlCbckH0YbVmo94QREk1S/view?usp=drive_link) |
| **LPFP**(PTv1*) | &cross; | 4 | 92.0% | 78.7% | 73.5% | [link](cfgs/lpfp_pcln/lpfp_ptnet_setmodify_t.yaml) | [link](https://drive.google.com/file/d/1ngERhbgub6Ewu7evgQyP-Ap5M4yXM5Xb/view?usp=drive_link) |
| **PCLN**(**LPFP**(PTv1*_0.5x)) | &cross; | 4 | 91.5% | 77.6% | 72.0% | [link](cfgs/lpfp_pcln/pcln_ptnet_setmodify.yaml) | [link](https://drive.google.com/file/d/1kmcE7KkZIZAqNGifaS8G1pDSQIVnFgiY/view?usp=drive_link) |
| **SPG**(PTv1) | &cross; | 2 | 91.2% | 77.9% | 71.5% | [link](cfgs/spg/spg_ptv1.yaml) | [link](https://drive.google.com/file/d/1ln76kOl6bdqQHjrLHFpZHYWWV885w3E_/view?usp=drive_link) |
| **SPG**(PTv2) | &cross; | 4 | 91.9% | 79.5% | 73.3% | [link](cfgs/spg/spg_ptv2.yaml) | [link](https://drive.google.com/file/d/1E3F1mAT1wqYMzKT7sLj8pD4KqtK3By3A/view?usp=drive_link) |


## Acknowledgment
_PointLiBR_ is developed by _[Jiawei Han](https://github.com/Javion11)_. It is derived from [PointNeXt](https://github.com/guochengqian/PointNeXt.git) and inspirited by several repos, e.g., [Pointcept](https://github.com/Pointcept/Pointcept.git), [PointNet++](https://github.com/charlesq34/pointnet2), [SPVNAS](https://github.com/mit-han-lab/spvnas.git), [RandLA-Net](https://github.com/QingyongHu/RandLA-Net.git). I am very grateful to the pioneers for their excellent work and open source spirit. ヾ(*´ー`*)ﾉ゛


## Citation
If you find _PointLiBR_ useful to your research, please cite my work as encouragement. (੭ˊ꒳​ˋ)੭✧
```
@article{han2024large,
  title={A Large-Scale Network Construction and Lightweighting Method for Point Cloud Semantic Segmentation},
  author={Han, Jiawei and Liu, Kaiqi and Li, Wei and Chen, Guangzhi and Wang, Wenguang and Zhang, Feng},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}

@inproceedings{han2024subspace,
  title={Subspace prototype guidance for mitigating class imbalance in point cloud semantic segmentation},
  author={Han, Jiawei and Liu, Kaiqi and Li, Wei and Chen, Guangzhi},
  booktitle={European Conference on Computer Vision},
  pages={255--272},
  year={2024},
  organization={Springer}
}

@ARTICLE{10933588,
  author={Han, Jiawei and Liu, Kaiqi and Li, Wei and Zhang, Feng and Xia, Xiang-Gen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Generating Inverse Feature Space for Class Imbalance in Point Cloud Semantic Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-17},
  keywords={Point cloud compression;Semantic segmentation;Training;Prototypes;Convolution;Transformers;Propagation losses;Data models;Cognition;Training data;Point Cloud Semantic Segmentation;Class Imbalance;Inverse Feature Space;Momentum-updated Prototypes;Dynamic Loss Weights},
  doi={10.1109/TPAMI.2025.3553051}}
```
