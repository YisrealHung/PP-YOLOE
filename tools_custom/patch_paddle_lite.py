# -*- coding: utf-8 -*-
"""
PP-YOLOE-Lite-Train: PaddleDetection Acceleration Patch
1. Injects RAM Byte-Cache into ppdet.data.transform.operators.Decode
   - Caches compressed JPEG/PNG bytes in memory (~500MB for 3,500 images)
   - Bypasses disk I/O on epochs 2~N, dropping data_cost to 0.001s
2. Injects NumPy 2.0 compatibility shim into ppdet/__init__.py
"""

import os
from pathlib import Path

BYTE_CACHE_PATCH_CODE = '''# --- [PP-YOLOE-Lite-Train RAM Byte-Cache Patch] ---
_GLOBAL_BYTE_CACHE = {}

def _patched_decode_apply(self, sample, context=None):
    if 'image' not in sample:
        im_file = sample.get('im_file')
        if im_file:
            raw_bytes = _GLOBAL_BYTE_CACHE.get(im_file)
            if raw_bytes is None:
                with open(im_file, 'rb') as f:
                    raw_bytes = f.read()
                _GLOBAL_BYTE_CACHE[im_file] = raw_bytes
            sample['image'] = raw_bytes
            sample.pop('im_file', None)

    try:
        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)
        if 'keep_ori_im' in sample and sample.get('keep_ori_im'):
            sample['ori_image'] = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception:
        im = sample['image']

    sample['image'] = im
    if 'h' not in sample:
        sample['h'] = im.shape[0]
    if 'w' not in sample:
        sample['w'] = im.shape[1]
    return sample

Decode.apply = _patched_decode_apply
# --- [End RAM Byte-Cache Patch] ---
'''

NUMPY_SHIM_CODE = '''import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }
for _k, _v in [("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str), ("alltrue", np.all)]:
    if not hasattr(np, _k):
        setattr(np, _k, _v)
'''

def apply_patches(repo_path='/content/PaddleDetection'):
    repo = Path(repo_path)
    if not repo.exists():
        print(f"[Warning] PaddleDetection repo not found at: {repo_path}")
        return False

    # 1. NumPy 2.0 Shim
    init_path = repo / 'ppdet' / '__init__.py'
    if init_path.exists():
        text = init_path.read_text(encoding='utf-8')
        if 'np.sctypes' not in text:
            init_path.write_text(NUMPY_SHIM_CODE + '\n' + text, encoding='utf-8')
            print("[Patch 1/2] Injected NumPy 2.0 compatibility shim into ppdet/__init__.py")
        else:
            print("[Patch 1/2] NumPy 2.0 shim already present.")

    # 2. RAM Byte-Cache in operators.py
    op_path = repo / 'ppdet' / 'data' / 'transform' / 'operators.py'
    if op_path.exists():
        text = op_path.read_text(encoding='utf-8')
        if '_GLOBAL_BYTE_CACHE' not in text:
            op_path.write_text(text + '\n\n' + BYTE_CACHE_PATCH_CODE, encoding='utf-8')
            print("[Patch 2/2] Injected RAM Byte-Cache into ppdet/data/transform/operators.py")
        else:
            print("[Patch 2/2] RAM Byte-Cache already present in operators.py")
    else:
        print(f"[Warning] operators.py not found at: {op_path}")

    return True

if __name__ == '__main__':
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else '/content/PaddleDetection'
    apply_patches(target)
