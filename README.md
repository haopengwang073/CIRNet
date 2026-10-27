# CIRNet (RA-L 2026)

**CIRNet: Compact Iterative Refinement Network for Real-time Stereo Matching** [![IEEE Xplore](https://img.shields.io/badge/IEEE-Paper-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11560925)

Haopeng Wang, Zhanhong Chen, Yu Zhang, Yanbiao Sun, and Jigui Zhu


## Installation

Create and activate a conda environment:

   ```bash
   conda create -n cirnet python=3.12 -y
   conda activate cirnet
   ```

Install PyTorch 2.8.0 and torchvision 0.23.0 with CUDA 12.8:

   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
   ```

Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```



## Checkpoints

Download the checkpoints from [Hugging Face](https://huggingface.co/haopengwang073/CIRNet)
and place them in `weights/`:

| Model | Link |
| --- | --- | 
| SceneFlow generalization | [general_sceneflow.pth](https://huggingface.co/haopengwang073/CIRNet/resolve/main/general_sceneflow.pth?download=true) |
| Million-scaling generalization | [general_million_scaling.pth](https://huggingface.co/haopengwang073/CIRNet/resolve/main/general_million_scaling.pth?download=true) | 



## Evaluation

Supply the root of the selected dataset. Expected layouts:

```text
ETH3D/
  two_view_training/<scene>/{im0.png,im1.png}
  two_view_training_gt/<scene>/{disp0GT.pfm,mask0nocc.png}
Middlebury/
  MiddEval3/trainingQ/<scene>/{im0.png,im1.png,disp0GT.pfm,mask0nocc.png}
KITTI_2012/
  training/{colored_0,colored_1,disp_occ}/*_10.png
KITTI_2015/
  training/{image_2,image_3,disp_occ_0}/*_10.png
```

Evaluation use stereo_evaluate.py

```bash
python stereo_evaluate.py --dataset eth3d --root /path/to/ETH3D \
  --checkpoint weights/general_million_scaling.pth --device cuda \
  --output outputs/eth3d.json

python stereo_evaluate.py --dataset middlebury --root /path/to/Middlebury --resolution Q \
  --checkpoint weights/general_million_scaling.pth --device cuda \
  --output outputs/middlebury_q.json

python stereo_evaluate.py --dataset kitti2012 --root /path/to/KITTI_2012 \
  --checkpoint weights/general_million_scaling.pth --device cuda \
  --output outputs/kitti2012.json

python stereo_evaluate.py --dataset kitti2015 --root /path/to/KITTI_2015 \
  --checkpoint weights/general_million_scaling.pth --device cuda \
  --output outputs/kitti2015.json
```

## Inference

```bash
python stereo_infer.py \
  --checkpoint weights/general_million_scaling.pth \
  --left /path/to/im0.png --right /path/to/im1.png \
  --output outputs/example --device cuda --iterations 4
```

The output directory contains:

- `disparity.npy`: float32 `[H, W]` disparity in original-image pixels.
- `disparity.pfm`: the same disparity, stored as a single-channel floating-point PFM.
- `disparity.png`: per-image min/max normalized grayscale preview, not metric disparity.







