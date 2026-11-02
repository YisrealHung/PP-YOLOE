# -*- coding: utf-8 -*-
"""
TensorRT Inference Utilities for PP-YOLOE on NVIDIA Jetson
Zero-Plugin Direct TensorRT Inference (exclude_nms=True)
"""

import cv2
import numpy as np

def preprocess(img_bgr, img_size):
    """Resize, BGR to RGB, Normalize by ImageNet mean & std"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    if (w, h) != (img_size, img_size):
        resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = rgb
    norm = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (norm - mean) / std
    norm = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[None, :], dtype=np.float32)
    scale_factor = np.array([[float(img_size) / h, float(img_size) / w]], dtype=np.float32)
    return norm, scale_factor

def postprocess(pred_bboxes, pred_scores, conf_thresh=0.35, iou_thresh=0.45):
    """
    PP-YOLOE+ TensorRT Postprocess (Coordinates already scaled back to original image size)
    pred_bboxes: [1, N, 4] -> (x1, y1, x2, y2)
    pred_scores: [1, C, N] -> Class scores
    """
    bboxes = pred_bboxes[0]
    scores = pred_scores[0].T  # [N, C]
    
    max_scores = np.max(scores, axis=-1)
    class_ids = np.argmax(scores, axis=-1)
    
    mask = max_scores >= conf_thresh
    filtered_boxes = bboxes[mask]
    filtered_scores = max_scores[mask]
    filtered_cls = class_ids[mask]
    
    if len(filtered_boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
        
    boxes_wh = filtered_boxes.copy()
    boxes_wh[:, 2] = boxes_wh[:, 2] - boxes_wh[:, 0]
    boxes_wh[:, 3] = boxes_wh[:, 3] - boxes_wh[:, 1]
    
    indices = cv2.dnn.NMSBoxes(boxes_wh.tolist(), filtered_scores.tolist(), conf_thresh, iou_thresh)
    keep = [idx if isinstance(idx, int) else idx[0] for idx in indices]
    return filtered_boxes[keep], filtered_scores[keep], filtered_cls[keep]

def draw(img_bgr, boxes, scores, cls_ids, class_names):
    vis = img_bgr.copy()
    for b, s, c in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, b)
        name = class_names[c] if c < len(class_names) else str(c)
        label = f"{name} {s:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return vis

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    class TRTEngine:
        def __init__(self, engine_path):
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.stream = cuda.Stream()

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(dev_mem))
                item = {'name': name, 'shape': shape, 'dtype': dtype, 'host': host_mem, 'dev': dev_mem}
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(item)
                else:
                    self.outputs.append(item)

        def infer(self, img_tensor, scale_factor_tensor):
            np.copyto(self.inputs[0]['host'], img_tensor.ravel())
            np.copyto(self.inputs[1]['host'], scale_factor_tensor.ravel())
            for item in self.inputs:
                cuda.memcpy_htod_async(item['dev'], item['host'], self.stream)
            for item in self.inputs + self.outputs:
                self.context.set_tensor_address(item['name'], int(item['dev']))
            self.context.execute_async_v3(self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['dev'], self.stream)
            self.stream.synchronize()
            return [out['host'].reshape(out['shape']) for out in self.outputs]
except ImportError:
    pass
