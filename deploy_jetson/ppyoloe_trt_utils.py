# -*- coding: utf-8 -*-
"""
TensorRT Inference Utilities for PP-YOLOE on NVIDIA Jetson
Zero-Plugin Direct TensorRT Inference (exclude_nms=True)
Uses PyTorch (torch.cuda) for seamless CUDA memory management!
Fallback to cuda-python or pycuda if torch is not installed.
"""

import cv2
import numpy as np
import tensorrt as trt

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


# ==============================================================================
# TensorRT Engine 執行類 (首選 PyTorch，次選 cuda-python / pycuda)
# ==============================================================================
BACKEND = None

# 第一優先順位：PyTorch (多數 Jetson 深度學習環境均已內建)
try:
    import torch
    if torch.cuda.is_available():
        BACKEND = 'torch'
except ImportError:
    pass

# 第二優先順位：cuda-python
if BACKEND is None:
    try:
        from cuda import cudart
        BACKEND = 'cuda_python'
    except ImportError:
        pass

# 第三優先順位：pycuda
if BACKEND is None:
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        BACKEND = 'pycuda'
    except ImportError:
        pass


class TRTEngine:
    def __init__(self, engine_path):
        if BACKEND is None:
            raise ImportError(
                "\n❌ 未檢測到可用的 CUDA 記憶體管理模組！\n"
                "請確認板端已安裝下列任一種套件：\n"
                "  - PyTorch: import torch; torch.cuda.is_available() (推薦)\n"
                "  - 或 cuda-python: pip3 install cuda-python\n"
                "  - 或 pycuda: pip3 install pycuda\n"
            )

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"無法載入 TensorRT Engine: {engine_path}"
        self.context = self.engine.create_execution_context()
        self.backend = BACKEND
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        num_tensors = self.engine.num_io_tensors if hasattr(self.engine, 'num_io_tensors') else len(self.engine)

        # ---------------------------------------------------------
        # 1. 首選：使用 PyTorch 進行 CUDA 顯存配置與記憶體管理
        # ---------------------------------------------------------
        if self.backend == 'torch':
            self.stream = torch.cuda.Stream()
            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i) if hasattr(self.engine, 'get_tensor_name') else self.engine[i]
                shape = tuple(self.engine.get_tensor_shape(name) if hasattr(self.engine, 'get_tensor_shape') else self.engine.get_binding_shape(i))
                dtype_trt = self.engine.get_tensor_dtype(name) if hasattr(self.engine, 'get_tensor_dtype') else self.engine.get_binding_dtype(i)
                dtype_np = trt.nptype(dtype_trt)
                dtype_torch = torch.from_numpy(np.empty(0, dtype=dtype_np)).dtype
                
                # 直接在 GPU 顯存分配 Tensor，免去 pycuda 複雜指針管理
                dev_tensor = torch.empty(shape, dtype=dtype_torch, device='cuda')
                item = {
                    'name': name,
                    'shape': shape,
                    'dtype': dtype_np,
                    'tensor': dev_tensor,
                    'ptr': dev_tensor.data_ptr()
                }
                self.bindings.append(dev_tensor.data_ptr())
                
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT if hasattr(self.engine, 'get_tensor_mode') else self.engine.binding_is_input(i)
                if is_input:
                    self.inputs.append(item)
                else:
                    self.outputs.append(item)
                    
            print(f"⚡ [TensorRT] 引擎載入成功！使用 PyTorch ({torch.__version__}) 管理 GPU 顯存")

        # ---------------------------------------------------------
        # 2. 次選：cuda-python
        # ---------------------------------------------------------
        elif self.backend == 'cuda_python':
            from cuda import cudart
            _, self.stream = cudart.cudaStreamCreate()
            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i) if hasattr(self.engine, 'get_tensor_name') else self.engine[i]
                shape = self.engine.get_tensor_shape(name) if hasattr(self.engine, 'get_tensor_shape') else self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name) if hasattr(self.engine, 'get_tensor_dtype') else self.engine.get_binding_dtype(i))
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                
                _, dev_ptr = cudart.cudaMalloc(size)
                host_buf = np.empty(shape, dtype=dtype)
                item = {'name': name, 'shape': shape, 'dtype': dtype, 'size': size, 'host': host_buf, 'ptr': int(dev_ptr)}
                self.bindings.append(int(dev_ptr))
                
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT if hasattr(self.engine, 'get_tensor_mode') else self.engine.binding_is_input(i)
                if is_input:
                    self.inputs.append(item)
                else:
                    self.outputs.append(item)
            print(f"⚡ [TensorRT] 引擎載入成功！使用原廠 cuda-python 管理 GPU 顯存")

        # ---------------------------------------------------------
        # 3. 備用：pycuda
        # ---------------------------------------------------------
        elif self.backend == 'pycuda':
            import pycuda.driver as cuda
            self.stream = cuda.Stream()
            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i) if hasattr(self.engine, 'get_tensor_name') else self.engine[i]
                shape = self.engine.get_tensor_shape(name) if hasattr(self.engine, 'get_tensor_shape') else self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name) if hasattr(self.engine, 'get_tensor_dtype') else self.engine.get_binding_dtype(i))
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                item = {'name': name, 'shape': shape, 'dtype': dtype, 'host': host_mem, 'ptr': int(dev_mem), 'dev_mem': dev_mem}
                self.bindings.append(int(dev_mem))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT if hasattr(self.engine, 'get_tensor_mode') else self.engine.binding_is_input(i)
                if is_input:
                    self.inputs.append(item)
                else:
                    self.outputs.append(item)
            print(f"⚡ [TensorRT] 引擎載入成功！使用 pycuda 管理 GPU 顯存")

    def infer(self, img_tensor, scale_factor_tensor):
        # ---------------------------------------------------------
        # PyTorch 推論：極速拷貝與非同步執行
        # ---------------------------------------------------------
        if self.backend == 'torch':
            with torch.cuda.stream(self.stream):
                # 將 NumPy 陣列拷貝至已在 GPU 上的 Tensor (zero-allocation)
                self.inputs[0]['tensor'].copy_(torch.from_numpy(img_tensor), non_blocking=True)
                self.inputs[1]['tensor'].copy_(torch.from_numpy(scale_factor_tensor), non_blocking=True)
                
                if hasattr(self.context, 'execute_async_v3'):
                    for item in self.inputs + self.outputs:
                        self.context.set_tensor_address(item['name'], item['ptr'])
                    self.context.execute_async_v3(self.stream.cuda_stream)
                else:
                    self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
                    
            self.stream.synchronize()
            return [out['tensor'].cpu().numpy() for out in self.outputs]

        # ---------------------------------------------------------
        # cuda-python 推論
        # ---------------------------------------------------------
        elif self.backend == 'cuda_python':
            from cuda import cudart
            np.copyto(self.inputs[0]['host'], img_tensor)
            np.copyto(self.inputs[1]['host'], scale_factor_tensor)
            
            for inp in self.inputs:
                cudart.cudaMemcpyAsync(inp['ptr'], inp['host'].ctypes.data, inp['size'], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
                
            if hasattr(self.context, 'execute_async_v3'):
                for item in self.inputs + self.outputs:
                    self.context.set_tensor_address(item['name'], item['ptr'])
                self.context.execute_async_v3(self.stream)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
                
            for out in self.outputs:
                cudart.cudaMemcpyAsync(out['host'].ctypes.data, out['ptr'], out['size'], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            return [out['host'].copy() for out in self.outputs]

        # ---------------------------------------------------------
        # pycuda 推論
        # ---------------------------------------------------------
        elif self.backend == 'pycuda':
            import pycuda.driver as cuda
            np.copyto(self.inputs[0]['host'], img_tensor.ravel())
            np.copyto(self.inputs[1]['host'], scale_factor_tensor.ravel())
            for item in self.inputs:
                cuda.memcpy_htod_async(item['dev_mem'], item['host'], self.stream)
            if hasattr(self.context, 'execute_async_v3'):
                for item in self.inputs + self.outputs:
                    self.context.set_tensor_address(item['name'], item['ptr'])
                self.context.execute_async_v3(self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['dev_mem'], self.stream)
            self.stream.synchronize()
            return [out['host'].reshape(out['shape']) for out in self.outputs]
