#!/bin/bash
# ==============================================================================
# NVIDIA Jetson Orin NX / Nano - PP-YOLOE-Lite TensorRT Build Script
# ==============================================================================
ONNX_FILE="${1:-ppyoloe_plus_s_640_trt_no_nms.onnx}"
ENGINE_FILE="${2:-ppyoloe_plus_s_640.engine}"
PRECISION="${3:-fp16}"
WORKSPACE="${4:-2048}"

echo "=========================================================="
echo "  Building TensorRT Engine on NVIDIA Jetson"
echo "  Input ONNX:    $ONNX_FILE"
echo "  Output Engine: $ENGINE_FILE"
echo "  Precision:     $PRECISION"
echo "  Workspace:     $WORKSPACE MB"
echo "=========================================================="

if [ ! -f "$ONNX_FILE" ]; then
    echo "❌ Error: ONNX file not found: $ONNX_FILE"
    exit 1
fi

TRTEXEC=/usr/src/tensorrt/bin/trtexec
if [ ! -f "$TRTEXEC" ]; then
    TRTEXEC=$(which trtexec)
fi

if [ -z "$TRTEXEC" ]; then
    echo "❌ Error: trtexec not found! Please check TensorRT installation."
    exit 1
fi

PRECISION_FLAG="--fp16"
if [ "$PRECISION" == "int8" ]; then
    PRECISION_FLAG="--int8"
elif [ "$PRECISION" == "fp32" ]; then
    PRECISION_FLAG=""
fi

$TRTEXEC --onnx="$ONNX_FILE" \
         --saveEngine="$ENGINE_FILE" \
         $PRECISION_FLAG \
         --memPoolSize=workspace:$WORKSPACE \
         --verbose=false

if [ $? -eq 0 ]; then
    echo "✅ TensorRT Engine built successfully: $ENGINE_FILE"
else
    echo "❌ Engine build failed. Please check log messages."
    exit 1
fi
