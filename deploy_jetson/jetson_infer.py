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

def parse_args():
    parser = argparse.ArgumentParser(description='Jetson PP-YOLOE+ TensorRT Inference')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT .engine file')
    parser.add_argument('--source', type=str, default='0', help='Image/dir path, video file, or camera index (default 0)')
    parser.add_argument('--labels', type=str, default='labels.txt', help='Path to labels.txt')
    parser.add_argument('--info', type=str, default='model_info.json', help='Path to model_info.json')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold (default 0.35)')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold (default 0.45)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Directory to save output images')
    parser.add_argument('--show', action='store_true', help='Display real-time window')
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

    print(f"🚀 Loading TensorRT Engine: {args.engine}")
    engine = U.TRTEngine(args.engine)
    os.makedirs(args.save_dir, exist_ok=True)

    is_cam = args.source.isdigit()
    is_video = not is_cam and args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_dir = os.path.isdir(args.source)

    if is_cam or is_video:
        src = int(args.source) if is_cam else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"❌ Failed to open source: {args.source}")
            return
        print("🎥 Streaming started (Press 'q' to quit)...")
        fps_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            inp, sf = U.preprocess(frame, img_size)
            outs = engine.infer(inp, sf)
            boxes, scores, cls_ids = U.postprocess(outs[0], outs[1], conf_thresh=args.conf, iou_thresh=args.iou)
            t1 = time.perf_counter()
            fps = 1.0 / max(1e-5, (t1 - t0))
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)

            vis = U.draw(frame, boxes, scores, cls_ids, class_names)
            cv2.putText(vis, f"FPS: {avg_fps:.1f} | Detections: {len(boxes)}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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
        t0 = time.perf_counter()
        inp, sf = U.preprocess(frame, img_size)
        outs = engine.infer(inp, sf)
        boxes, scores, cls_ids = U.postprocess(outs[0], outs[1], conf_thresh=args.conf, iou_thresh=args.iou)
        dt = (time.perf_counter() - t0) * 1000
        vis = U.draw(frame, boxes, scores, cls_ids, class_names)
        save_p = os.path.join(args.save_dir, 'result.jpg')
        cv2.imwrite(save_p, vis)
        print(f"✅ Inference complete: {dt:.2f} ms, detected {len(boxes)} objects, saved to {save_p}")

if __name__ == '__main__':
    main()

