# -*- coding: utf-8 -*-
"""
Dynamic Configuration Generator for PP-YOLOE-Lite-Train
Applies architectural and pipeline optimizations borrowed from PP-YOLOE-R:
1. static_assigner_epoch: -1 (Bypasses ATSS assigner, pure TAL from Epoch 0)
2. Fixed-Scale Single Resolution (Locks cuDNN convolution algorithm caching)
3. Streamlined Data Pipeline (Removes RandomDistort and RandomExpand)
4. PadGT preservation (Ensures pad_gt_mask is generated for TaskAlignedAssigner)
"""

import os
from pathlib import Path

SCALE_CONFIGS = {
    's': {
        'depth': 0.33,
        'width': 0.50,
        'url': 'https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams'
    },
    'm': {
        'depth': 0.67,
        'width': 0.75,
        'url': 'https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_m_obj365_pretrained.pdparams'
    },
    'l': {
        'depth': 1.00,
        'width': 1.00,
        'url': 'https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_l_obj365_pretrained.pdparams'
    },
    'x': {
        'depth': 1.33,
        'width': 1.25,
        'url': 'https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_plus_crn_x_80e_coco.pdparams'
    }
}

def make_lite_config(
    model_scale='s',
    num_classes=80,
    dataset_dir='/content/dataset/coco',
    img_size=640,
    epochs=50,
    batch_size=32,
    base_lr=None,
    eval_interval=5,
    pretrain_weights=None,
    output_path=None
):
    assert model_scale in SCALE_CONFIGS, f"Invalid model_scale: {model_scale}. Choose from {list(SCALE_CONFIGS.keys())}"
    sc = SCALE_CONFIGS[model_scale]
    
    if pretrain_weights is None:
        pretrain_weights = sc['url']
        
    if base_lr is None:
        # Linear scaling rule from base batch size 8 (lr=0.00125)
        base_lr = round(0.00125 * (batch_size / 8.0), 5)

    content = f"""_BASE_: [
  '../runtime.yml',
  './_base_/optimizer_80e.yml',
  './_base_/ppyoloe_plus_crn.yml',
  './_base_/ppyoloe_plus_reader.yml'
]

log_iter: 10
snapshot_epoch: {eval_interval}
weights: output/ppyoloe_lite_{model_scale}_custom/model_final
pretrain_weights: {pretrain_weights}
depth_mult: {sc['depth']}
width_mult: {sc['width']}

norm_type: bn

metric: COCO
num_classes: {num_classes}

# 🚀 PP-YOLOE-R 核心加速 1: 標籤分配器全面優化
# 將 static_assigner_epoch 設為 -1，徹底跳過計算繁重的 ATSSAssigner，從 Epoch 0 起直接使用 TaskAlignedAssigner
PPYOLOEHead:
  static_assigner_epoch: -1

TrainDataset:
  !COCODataSet
    image_dir: train
    anno_path: annotations/train.json
    dataset_dir: {dataset_dir}
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: val
    anno_path: annotations/val.json
    dataset_dir: {dataset_dir}

TestDataset:
  !ImageFolder
    anno_path: {dataset_dir}/annotations/val.json
    dataset_dir: {dataset_dir}

epoch: {epochs}
LearningRate:
  base_lr: {base_lr}
  schedulers:
  - !CosineDecay
    max_epochs: {int(epochs * 1.2)}
    min_lr_ratio: 0.05
  - !LinearWarmup
    start_factor: 0.0
    epochs: 3

worker_num: 4
eval_height: &eval_height {img_size}
eval_width: &eval_width {img_size}
eval_size: &eval_size [*eval_height, *eval_width]

# 🚀 PP-YOLOE-R 核心加速 2: 輕量流線型資料讀取器 (Streamlined Fast Reader)
# 1. 徹底拔除 RandomDistort (每張圖4次CPU像素循環) 與 RandomExpand (4倍畫布顯存拷貝)
# 2. 徹底拔除 BatchRandomResize 多尺度隨機縮放，固定為單一尺寸，釋放 cuDNN 靜態卷積核心性能
# 3. 保留 PadGT 以產生 pad_gt_mask，保證 TaskAlignedAssigner 計算正確
TrainReader:
  sample_transforms:
    - Decode: {{}}
    - RandomFlip: {{}}
    - Resize:
        target_size: [*eval_height, *eval_width]
        keep_ratio: False
        interp: 1
  batch_transforms:
    - NormalizeImage: {{mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}}
    - Permute: {{}}
    - PadGT: {{}}
  batch_size: {batch_size}
  shuffle: true
  drop_last: true
  use_shared_memory: true
  collate_batch: true

EvalReader:
  sample_transforms:
    - Decode: {{}}
    - Resize:
        target_size: *eval_size
        keep_ratio: False
        interp: 1
    - NormalizeImage: {{mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}}
    - Permute: {{}}
  batch_size: 2
"""
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.lstrip())
        print(f"[Config] Generated PP-YOLOE-Lite config at: {output_path}")

    return content

if __name__ == '__main__':
    cfg = make_lite_config(model_scale='s', num_classes=1, output_path='configs/ppyoloe_plus_crn_s_lite.yml')
