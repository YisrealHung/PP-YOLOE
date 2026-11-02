# -*- coding: utf-8 -*-
"""
Jetson Orin NX / Nano - Real-Time TensorRT Inference Tool
Supports:
- Single image
- Image directory
- Video file
- Live USB / CSI camera (with real-time FPS counter)
"""

import argparse
import json
import os
import time
import cv2
import numpy as np
import ppyoloe_trt_utils as U

# 抑制 OpenCV/libjpeg 警告訊息
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

def parse_args():
    parser = argparse.ArgumentParser(description='Jetson PP-YOLOE+ TensorRT Inference')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT .engine file')
    parser.add_argument('--source', type=str, default='0', help='Image/dir path, video file, or camera index (default 0)')
    parser.add_argument('--labels', type=str, default='labels.txt', help='Path to labels.txt')
    parser.add_argument('--info', type=str, default='model_info.json', help='Path to model_info.json')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold (default 0.35)')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold (default 0.45)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Directory to save output images')
    parser.add_argument('--benchmark', action='store_true', help='Print detailed stage-by-stage timing profile')
    parser.add_argument('--cuda-pre', action='store_true', help='Enable PyTorch GPU accelerated preprocessing (much faster)')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations for accurate benchmark (default 3)')
    parser.add_argument('--show', action='store_true', help='Display real-time window')
    parser.add_argument('--cam-fps', type=int, default=30, help='Camera requested FPS (default 30)')
    parser.add_argument('--cam-width', type=int, default=1280, help='Camera width (default 1280)')
    parser.add_argument('--cam-height', type=int, default=720, help='Camera height (default 720)')
    return parser.parse_args()

def main():
    args = parse_args()
    if os.path.exists(args.info):
        with open(args.info, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        img_size = model_info.get('img_size', 640)
        class_names = model_info.get('class_names', [])
    elif os.path.exists(args.labels):
        img_size = 640
        class_names = [line.strip() for line in open(args.labels, encoding='utf-8') if line.strip()]
    else:
        img_size = 640
        class_names = []

    # 預處理函式選擇 (CPU vs GPU)
    preproc_fn = U.preprocess_gpu if (args.cuda_pre and U.BACKEND == 'torch') else U.preprocess
    pre_mode_name = "PyTorch GPU ⚡" if (args.cuda_pre and U.BACKEND == 'torch') else "CPU (OpenCV/NumPy)"
    print(f"🚀 Loading TensorRT Engine: {args.engine} (預處理模式: {pre_mode_name})")
    engine = U.TRTEngine(args.engine)
    os.makedirs(args.save_dir, exist_ok=True)

    is_cam = args.source.isdigit()
    is_video = not is_cam and args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_dir = os.path.isdir(args.source)

    if is_cam or is_video:
        src = int(args.source) if is_cam else args.source
        if is_cam:
            # 優先使用 V4L2 後端以消除 GStreamer 警告與協商延遲
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
            # 強制開啟 MJPG 格式以釋放 USB 頻寬，解鎖 30 FPS 硬體上限
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
            # 關閉緩衝區排隊 (避免延遲堆積)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"❌ Failed to open source: {args.source}")
            return
        print("🎥 Streaming started (Press 'q' to quit)...")
        stream_fps_list = []
        algo_time_list = []
        draw_time_list = []
        total_time_list = []
        
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            t_cap_end = time.perf_counter()
            
            # 1. 演算法推論階段 (預處理 + TRT 推論 + NMS 後處理)
            inp, sf = preproc_fn(frame, img_size)
            outs = engine.infer(inp, sf)
            boxes, scores, cls_ids = U.postprocess(outs[0], outs[1], conf_thresh=args.conf, iou_thresh=args.iou)
            t_algo_end = time.perf_counter()

            # 2. 繪製 BBox 階段
            vis = U.draw(frame, boxes, scores, cls_ids, class_names)
            t_draw_end = time.perf_counter()

            # 計算各階段耗時 (ms)
            dt_cap = (t_cap_end - t_frame_start) * 1000
            dt_algo = (t_algo_end - t_cap_end) * 1000
            dt_draw = (t_draw_end - t_algo_end) * 1000
            dt_total = (t_draw_end - t_frame_start) * 1000
            
            algo_time_list.append(dt_algo)
            draw_time_list.append(dt_draw)
            total_time_list.append(dt_total)
            stream_fps = 1.0 / max(1e-5, (t_draw_end - t_frame_start))
            stream_fps_list.append(stream_fps)
            
            if len(stream_fps_list) > 30:
                stream_fps_list.pop(0)
                algo_time_list.pop(0)
                draw_time_list.pop(0)
                total_time_list.pop(0)

            avg_fps = sum(stream_fps_list) / len(stream_fps_list)
            avg_algo = sum(algo_time_list) / len(algo_time_list)
            avg_draw = sum(draw_time_list) / len(draw_time_list)
            avg_total = sum(total_time_list) / len(total_time_list)

            # 在畫面上繪製即時詳細資訊
            cv2.putText(vis, f"Cam FPS: {avg_fps:.1f} | Algo: {avg_algo:.1f}ms (TRT+NMS) | Draw: {avg_draw:.1f}ms", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(vis, f"Total Pipeline: {avg_total:.1f}ms | Detections: {len(boxes)}", 
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # 終端機即時輸出各階段細項耗時 (每 30 幀更新一次)
            if len(stream_fps_list) % 30 == 0:
                print(f"⚡ [Stream FPS: {avg_fps:4.1f}] 演算法推論: {avg_algo:5.1f} ms | 繪製Bbox: {avg_draw:4.1f} ms | 總處理時間: {avg_total:5.1f} ms | 偵測數: {len(boxes)}")

            if args.show:
                cv2.imshow('Jetson PP-YOLOE-Lite', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
    elif is_dir:
        imgs = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📁 Processing directory {args.source} ({len(imgs)} images)...")
        for p in imgs:
            frame = cv2.imread(p)
            if frame is None:
                continue
            t0 = time.perf_counter()
            inp, sf = U.preprocess(frame, img_size)
            outs = engine.infer(inp, sf)
            boxes, scores, cls_ids = U.postprocess(outs[0], outs[1], conf_thresh=args.conf, iou_thresh=args.iou)
            dt = (time.perf_counter() - t0) * 1000
            vis = U.draw(frame, boxes, scores, cls_ids, class_names)
            save_p = os.path.join(args.save_dir, os.path.basename(p))
            cv2.imwrite(save_p, vis)
            print(f"{os.path.basename(p)} -> {len(boxes)} objects ({dt:.1f} ms)")
    else:
        frame = cv2.imread(args.source)
        assert frame is not None, f"Failed to read image: {args.source}"

        # 預熱 (Warmup) 消除首次 CUDA Kernel JIT/配置延遲
        if args.warmup > 0:
            dummy_inp, dummy_sf = preproc_fn(frame, img_size)
            for _ in range(args.warmup):
                engine.infer(dummy_inp, dummy_sf)

        t_total_start = time.perf_counter()

        # 1. 預處理 (Preprocess: Resize, BGR2RGB, Norm, Transpose)
        t_pre_start = time.perf_counter()
        inp, sf = preproc_fn(frame, img_size)
        t_pre_end = time.perf_counter()

        # 2. TensorRT 推論 (包含 Host->Device, GPU Kernel, Device->Host)
        outs, trt_times = engine.infer(inp, sf, benchmark=True)
        t_trt_end = time.perf_counter()

        # 3. 後處理 (Postprocess: NMS, Score Filtering, Box Scaling)
        boxes, scores, cls_ids = U.postprocess(outs[0], outs[1], conf_thresh=args.conf, iou_thresh=args.iou)
        t_post_end = time.perf_counter()

        # 4. 繪製與可視化 (Draw)
        vis = U.draw(frame, boxes, scores, cls_ids, class_names)
        t_draw_end = time.perf_counter()

        # 5. 檔案保存 (Save)
        save_p = os.path.join(args.save_dir, 'result.jpg')
        cv2.imwrite(save_p, vis)
        t_save_end = time.perf_counter()

        dt_pre = (t_pre_end - t_pre_start) * 1000
        dt_trt = (t_trt_end - t_pre_end) * 1000
        dt_post = (t_post_end - t_trt_end) * 1000
        dt_draw = (t_draw_end - t_post_end) * 1000
        dt_save = (t_save_end - t_draw_end) * 1000
        dt_pipeline = dt_pre + dt_trt + dt_post
        dt_total = (t_save_end - t_total_start) * 1000

        print(f"\n==================== ⏱️ 耗時細項分析 (Timing Profile) ====================")
        print(f" 1. 影像預處理 ({pre_mode_name})      : {dt_pre:6.2f} ms")
        print(f" 2. TensorRT 引擎推論 (TRT Total)      : {dt_trt:6.2f} ms")
        print(f"    ├─ 記憶體複製 CPU ➔ GPU (H2D)     : {trt_times['h2d']:6.2f} ms")
        print(f"    ├─ GPU 核心推論 (CUDA Kernel)     : {trt_times['gpu']:6.2f} ms  (純 GPU 計算)")
        print(f"    └─ 記憶體複製 GPU ➔ CPU (D2H)     : {trt_times['d2h']:6.2f} ms")
        print(f" 3. 後處理與 NMS (Postprocess, CPU)    : {dt_post:6.2f} ms")
        print(f" -------------------------------------------------------------------------")
        print(f" 🚀 演算法總推論 (Pre + TRT + Post)   : {dt_pipeline:6.2f} ms  (相當於 {1000.0/max(1e-5, dt_pipeline):.1f} FPS)")
        print(f" 4. 畫框與文字可視化 (Draw)            : {dt_draw:6.2f} ms")
        print(f" 5. 磁碟寫入儲存 (Disk Save)           : {dt_save:6.2f} ms")
        print(f" -------------------------------------------------------------------------")
        print(f" 🏁 完整生命週期耗時 (End-to-End Total) : {dt_total:6.2f} ms")
        print(f"==========================================================================")
        print(f"✅ 偵測到 {len(boxes)} 個物件，成果圖已保存至: {save_p}\n")

if __name__ == '__main__':
    main()

