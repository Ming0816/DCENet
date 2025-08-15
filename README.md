# DCENet: Dual-Branch Context Extractor for Multi-Scale Features in Image Deblurring



## Installation
This implementation is based on [BasicSR](https://github.com/xinntao/BasicSR) which is an open-source toolbox for image/video restoration tasks, [CascadedGaze](https://github.com/Ascend-Research/CascadedGaze.git), [NAFNet](https://github.com/megvii-research/NAFNet), [Restormer](https://github.com/swz30/Restormer/tree/main/Denoising) and [Multi Output Deblur](https://github.com/Liu-SD/multi-output-deblur)

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```



##  Single Image Deblurring on GoPro
### 1. Data Preparation
##### Download the train set (in disk format) and place it in ```./datasets/GoPro/train/```:

* [google drive](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing)
* it should be like ```./datasets/GoPro/train/input/ ``` and ```./datasets/GoPro/train/target/```
* ```python scripts/data_preparation/gopro.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.
* The final train data structure should be like ```./datasets/GoPro/train/blur_crops.lmdb/``` and ```./datasets/GoPro/train/sharp_crops.lmdb/```

##### Download the evaluation data (in lmdb format) and place it in ```./datasets/GoPro/test/```:

  * [google drive](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view?usp=sharing)
  * it should be like ```./datasets/GoPro/test/input.lmdb/``` and ```./datasets/GoPro/test/target.lmdb/```

```bash
./datasets/
└── GoPro/
    ├── train/
    │   ├── blur_crops.lmdb/
    │   └── sharp_crops.lmdb/
    └── test/
        ├── input.imdb/
        └── target.imdb/

```



### 2. Training

* To train the DCENet model with one NVIDIA V100 gpu:

```
torchrun --nproc_per_node=1 train.py -opt ./options/train/GoPro/DCENet-GoPro.yml --launcher pytorch
```

### 3. Evaluation


* Download the pre-trained [GoPro](https://drive.google.com/file/d/1XCcPRU1u1CbgpzqOiH3zBH5FGXlQBWFd/view?usp=drive_link) models and place them in ./DCENet/


* Download [HIDE](https://drive.google.com/file/d/1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A/view?usp=sharing), [RealBlurR](https://drive.google.com/file/d/1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS/view?usp=sharing), [RealBlurJ](https://drive.google.com/file/d/1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW/view?usp=sharing) test datasets and place them in `./datasets/`.

  * Testing on GoPro dataset:
```
torchrun --nproc_per_node=1 test.py -opt ./options/test/GoPro/DCENet-GoPro.yml --launcher pytorch
```
  
  * Testing on HIDE dataset:
```
torchrun --nproc_per_node=1 test.py -opt ./options/test/HIDE/DCENet-HIDE.yml --launcher pytorch
```
  
  * Testing on RealBlurR dataset:
```
torchrun --nproc_per_node=1 test.py -opt ./options/test/RealBlur_R/DCENet-RealBlur_R.yml --launcher pytorch
```

  * Testing on RealBlurJ dataset:
```
torchrun --nproc_per_node=1 test.py -opt ./options/test/RealBlur_J/DCENet-RealBlur_J.yml --launcher pytorch
```

##  Defocus Deblurring on DPDD
### 1. Data Preparation
* Download the [train](https://drive.google.com/file/d/1bl5i1cDQNvkgVA_x37QdhvvFk1R80kfe/view?usp=sharing), [val](https://drive.google.com/file/d/1KRAmBzluu-IG9-BOsuakB5rjY5_f-kiR/view?usp=sharing), [test](https://drive.google.com/file/d/1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw/view?usp=sharing) sets:
* Generate image patches from full-resolution training and val images:
```
python generate_patches_dpdd.py 
```
* Directory structure should look like this
```bash  
 ./datasets/Defocus_Deblurring/
 ├──train/
     └──DPDD/
          ├──inputL_crops/
          ├──inputR_crops/
          ├──inputC_crops/
          └──target_crops/
 ├──val/
     └──DPDD/
          ├──inputL_crops/
          ├──inputR_crops/
          ├──inputC_crops/
          └──target_crops/
 └──test/
     └──DPDD/
          ├──inputL/
          ├──inputR/
          ├──inputC/
          ├──target/
          ├──indoor_labels.npy
          └──outdoor_labels.npy
```

### 2. Training
  * To train on **single-image** defocus deblurring task with one NVIDIA V100 gpu:
```
torchrun --nproc_per_node=1 train.py -opt ./options/train/Defocus-Deblur/DefocusDeblur_Single_8bit_DCENet.yml --launcher pytorch
```
* To train on **dual-pixel** defocus deblurring task  with single gpu:
```
torchrun --nproc_per_node=1 train.py -opt ./options/train/Defocus-Deblur/DefocusDeblur_DualPixel_16bit_DCENet.yml --launcher pytorch
```

### 3. Evaluation

* Download the pre-trained [Single-image](https://drive.google.com/file/d/1gXHRheK_PrQRbwlKnlyhk9RFIn6ggze-/view?usp=drive_link) and [dual-pixel](https://drive.google.com/file/d/10879vSuKa1p2q9pCi7epI0xQd8SP0CeE/view?usp=drive_link) models and place them in ./DCENet/

 * Testing on **single-image** defocus deblurring task, run
```
python test_single_image_defocus_deblur.py --save_images
```

 * Testing on **dual-pixel** defocus deblurring task, run
```
python test_dual_pixel_defocus_deblur.py --save_images
```
