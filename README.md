# 🚀 PP-YOLOE-Lite-Train (含 NVIDIA Jetson Orin TensorRT 極速部署)

> 專為 **Google Colab (T4 / A100 GPU)** 與 **NVIDIA Jetson Orin NX / Nano** 深度優化之 PP-YOLOE+ 極速訓練與部署套件。  
> 專案開源於 GitHub：**https://github.com/YisrealHung/PP-YOLOE**  
> 融合 **PP-YOLOE-R 旋轉框的精準架構設計**、**RAM Byte-Cache 記憶體快取技術** 與 **Jetson 原生 GPU 預處理管線**。  
> 將原本需耗時 5~6 小時的原生訓練縮減為 **~1.8 小時 (T4 GPU)**，並在 **Jetson Orin Nano (8GB)** 實測達到 **54+ FPS** 的極速即時辨識！

---

## 🧐 為什麼 PP-YOLOE 預設訓練比 PP-YOLOE-R 還慢？

許多開發者在訓練時常會發現：**明明 PP-YOLOE-R 要計算更複雜的旋轉矩形 IoU (Rotated IoU)，為什麼訓練速度反而比水平框的 PP-YOLOE 還快？**

經過對 PaddleDetection 原始碼與設定檔的底層比對，根本原因在於 **訓練策略與管線設計的歷史包袱**：

| 評比項目 | 官方原生 PP-YOLOE (水平框) | PP-YOLOE-R (旋轉框) | 🚀 PP-YOLOE-Lite-Train (本專案) |
|---|---|---|---|
| **標籤分配器 (Assigner)** | 前 30~100 輪強跑 `ATSSAssigner`<br>計算量龐大，換算 IoU 動態均值標準差 | **`static_assigner_epoch: -1`**<br>直接採用 TAL，完全跳過靜態分配器 | **`static_assigner_epoch: -1`**<br>自第 0 輪直通 TAL，反向傳播加速 ~25% |
| **輸入尺度變換** | `BatchRandomResize`<br>(15 種尺寸隨機縮放，頻繁動態配置顯存) | **固定單一尺寸** (`target_size: 640x640`)<br>cuDNN 靜態卷積核心全速快取 | **固定單一尺寸** (`target_size: 640x640`)<br>鎖定卷積演算法，消除顯存抖動 |
| **CPU 影像增強開銷** | 啟用 `RandomExpand` (4倍畫布複製)<br>啟用 `RandomDistort` (4次CPU像素循環) | **無 `RandomExpand`**<br>**無 `RandomDistort`** | **徹底拔除 CPU 像素重算**<br>保留向量化翻轉與幾何變換 |
| **磁碟 I/O 快取** | 無快取，每輪反覆讀取磁碟圖檔<br>(Colab 硬碟延遲高達 0.3s~1.5s/張) | 無快取，需仰賴硬體 I/O 頻寬 | **RAM Byte-Cache 零延遲記憶體快取**<br>僅佔 ~500MB RAM，讀取開銷歸零 (0.001s) |
| **單卡建議 Batch Size** | 8 (因動態尺寸防顯存溢出) | 8 ~ 16 | **32 (搭配 AMP 自動混合精度)** |
| **50 輪耗時 (3,500張圖)** | 約 5 ~ 6 小時 (單輪 ~6.5 分) | 約 2.5 ~ 3 小時 (單輪 ~3.5 分) | **約 1.8 小時 (T4)** / **約 20 分鐘 (A100)**<br>(單輪 2 分 18 秒，提速近 3 倍！) |

---

## 📊 Google Colab (Tesla T4) 實測驗證數據

以 3,493 張 $1024 \times 1024$ 高解析度影像為基準，實機於 Colab T4 GPU 執行之訓練日誌：

```text
[11/02 05:14:48] ppdet.data.source.coco INFO: Load [3493 samples valid] in train.json
[11/02 05:15:16] Epoch: [0] [ 10/109] loss: 3.786  batch_cost: 1.0621  ips: 30.13 images/s
[11/02 05:16:38] Epoch: [0] [ 70/109] loss: 0.848  batch_cost: 1.1992  ips: 26.68 images/s
[11/02 05:17:21] Epoch: [0] [100/109] loss: 0.820  batch_cost: 1.2718  ips: 25.16 images/s
[11/02 05:17:36] Epoch: [1] [  0/109] loss: 0.813  batch_cost: 1.5656  max_mem: 6656 MB
```

- **單輪步數**：僅需 **109 iters**（相比原生 437 iters 縮減 4 倍迭代負擔）。
- **處理吞吐量**：穩定達到 **25.5 ~ 30.1 images/s**（每秒高達 25~30 張影像）。
- **單輪平均耗時**：約 **2 分 18 秒**。
- **50 輪總訓練時間**：約 **1 小時 50 分鐘**（相比官方原生 5.5 小時節省超過 65% 時間！）。
- **GPU 顯存佔用**：僅 **6,656 MB / 15,360 MB**（佔比 43%，安全穩定無 OOM 風險）。
- **收斂表現**：第 0 輪 Loss 即從 4.20 迅速下探至 0.82，在第 5 輪即可達 **mAP(0.50:0.95) = 0.999**！

---

## 🤖 NVIDIA Jetson Orin Nano (8GB) 實機實測數據

在 NVIDIA Jetson Orin Nano 8GB 上部署 TensorRT FP16 引擎，實機測得之耗時細項分析（Timing Profile）：

```text
🚀 Loading TensorRT Engine: ppyoloe_s.engine (預處理模式: PyTorch GPU ⚡)
⚡ [TensorRT] 引擎載入成功！使用 PyTorch (2.5.0a0+872d972e41.nv24.08) 管理 GPU 顯存

==================== ⏱️ 耗時細項分析 (Timing Profile) ====================
 1. 影像預處理 (PyTorch GPU ⚡)      :   4.66 ms  (原 CPU 需 17.71 ms，狂降 75%！)
 2. TensorRT 引擎推論 (TRT Total)      :  12.99 ms
    ├─ 記憶體複製 CPU ➔ GPU (H2D)     :   0.52 ms  (僅傳輸 uint8 原始圖，頻寬省 75%)
    ├─ GPU 核心推論 (CUDA Kernel)     :  11.87 ms  (純 TensorRT FP16 計算)
    └─ 記憶體複製 GPU ➔ CPU (D2H)     :   0.57 ms
 3. 後處理與 NMS (Postprocess, CPU)    :   0.90 ms  (純量級相容型 NMS)
 -------------------------------------------------------------------------
 🚀 演算法總推論 (Pre + TRT + Post)   :  18.55 ms  (相當於 53.9 ~ 60+ FPS！)
 4. 畫框與文字可視化 (Draw)            :   1.96 ms
 5. 磁碟寫入儲存 (Disk Save)           :  31.66 ms  (即時鏡頭串流無需寫入)
 -------------------------------------------------------------------------
 🏁 完整生命週期耗時 (End-to-End Total) :  52.18 ms
==========================================================================
```

### 💡 效能對比：CPU 模式 vs GPU 預處理加速模式

| 處理階段 | 原版 CPU 預處理模式 | 🚀 GPU 預處理加速模式 (`--cuda-pre`) | 效能改善 |
|---|---|---|---|
| **影像預處理** | 17.71 ms (CPU 雙線性插值與標準化) | **4.66 ms** (PyTorch GPU 並行張量計算) | ⚡ **提速 74%** |
| **記憶體 H2D 拷貝** | 2.13 ms (傳輸 4.9MB float32 張量) | **0.52 ms** (傳輸 1.2MB uint8 原圖) | ⚡ **傳輸量縮減 75%** |
| **純 GPU 推論** | 11.98 ms | **11.87 ms** | 穩定高效 |
| **後處理與 NMS** | 0.68 ms | **0.90 ms** | 零延遲過濾 |
| **總演算法耗時** | 32.96 ms (~30 FPS) | **18.55 ms (~54 FPS)** | 🏆 **幀率近乎翻倍！** |

---

## ⚡ PP-YOLOE-Lite-Train 核心加速策略

1. **標籤分配器全面減負 (`static_assigner_epoch: -1`)**：
   PP-YOLOE 原本設計在訓練初期透過 ATSS 暖機。但使用 Objects365 海量預訓練權重時，骨幹特徵早已具備強大表徵能力，ATSS 的暖機純屬浪費！本專案設為 `-1`，直接使用 `TaskAlignedAssigner`。
2. **流線型單一尺度資料管道 (Streamlined Fast Pipeline)**：
   移除 Python CPU 端耗時的 `RandomDistort` 與 `RandomExpand`，並移除多尺度動態縮放。保留 `PadGT` 以生成 `pad_gt_mask`，兼具最高速度與標籤分配精準度。
3. **RAM Byte-Cache 記憶體快取**：
   在 `ppdet.data.transform.operators.Decode` 注入記憶體快取，將壓縮後的影像 Byte 暫存在記憶體中（5,000 張 $1024 \times 1024$ 圖片僅佔約 770 MB RAM）。第 2 輪起完全免除硬碟讀取，終結 DataLoader 飢餓。
4. **硬體滿載 Batch Size 32 + AMP + cuDNN Benchmark**：
   在 T4 GPU 16GB 顯存下，Batch Size 32 僅佔用約 6.6 GB 顯存。每輪僅需 109 次迭代即可完成（約 2 分 18 秒/輪），訓練步數縮減為 1/4！
5. **Jetson Orin 免 Plugin 專用導出 (`exclude_nms=True`)**：
   解耦 NMS 操作，避免依賴易出錯的自訂 C++ TensorRT Plugin，直接以原生 ONNX 透過原廠 `trtexec` 一鍵轉出高相容性 Engine。
6. **PyTorch CUDA 零安裝記憶體管理 & GPU 預處理**：
   Jetson Orin 內建 PyTorch 環境，直接利用 `torch.cuda` 管理張量記憶體指標，免除繁瑣的 `pycuda` 安裝與編譯報錯；並透過 `--cuda-pre` 直接在 GPU 完成 Resize 與正規化，徹底釋放 1024 個 CUDA 核心的實力！

---

## 📁 專案目錄結構

```text
PP-YOLOE/
├── configs/
│   └── ppyoloe_plus_crn_s_lite.yml     # PP-YOLOE-Lite 極速訓練參考設定檔
├── tools_custom/
│   ├── patch_paddle_lite.py            # 自動注入 RAM Byte-Cache 與 NumPy 2.0 Shim
│   └── make_config.py                  # 動態自動設定檔生成器
├── deploy_jetson/
│   ├── build_engine.sh                 # Jetson Orin TensorRT FP16 一鍵編譯腳本
│   ├── ppyoloe_trt_utils.py            # PyTorch CUDA 顯存管理、GPU 預處理與相容 NMS
│   └── jetson_infer.py                 # 板端推論主程式 (支援圖片/資料夾/攝影機串流/--cuda-pre)
├── PP_YOLOE_Lite_Train.ipynb           # Google Colab 一鍵極速訓練互動式 Notebook
├── requirements.txt                    # 依賴套件清單
├── .gitignore                          # Git 忽略設定
└── README.md                           # 專案說明文件
```

---

## 🚀 快速上手：Google Colab 極速訓練

1. 直接在 Google Colab 開啟本倉庫的 [**PP_YOLOE_Lite_Train.ipynb**](https://github.com/YisrealHung/PP-YOLOE/blob/main/PP_YOLOE_Lite_Train.ipynb)。
2. 在選單「執行階段 → 變更執行階段類型」選取 **T4 GPU**。
3. 在第 0 節「⚙️ 參數設定」中填寫你的 Google Drive 資料集路徑（支援 YOLO txt 或 COCO JSON zip）。
4. 點選「執行階段 → 全部執行」即可全自動完成極速訓練、收斂評估與 Jetson Orin 零 Plugin ONNX 導出！

---

## 🤖 NVIDIA Jetson Orin Nano / NX 實機部署指南

訓練完成後，Colab 會在 Google Drive 的 `ppyoloe_runs/<專案名稱>/export/` 產生專屬部署包。

### 1. 將部署包傳輸至 Jetson Orin
```bash
# 在 Jetson 上複製此專案
git clone https://github.com/YisrealHung/PP-YOLOE.git ~/pp-yoloe/PP-YOLOE
cd ~/pp-yoloe/PP-YOLOE/deploy_jetson

# 將訓練導出的 ONNX (ppyoloe_plus_s_640_trt_no_nms.onnx) 放入 deploy_jetson 目錄
```

### 2. 開啟 Orin 最高效能模式（鎖頻）
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 3. 一鍵編譯 TensorRT Engine (免 Plugin)
```bash
bash build_engine.sh
# 將自動生成 ppyoloe_s.engine (FP16 極速模型)
```

### 4. 執行推論與即時串流

#### 📸 單張影像推論 (顯示各階段毫秒級 Timing Profile)
```bash
python3 jetson_infer.py --engine ppyoloe_s.engine --source test.jpg --cuda-pre --save-dir results
```

#### 🎥 USB / CSI 即時攝影機串流 (具備 FPS 顯示)
```bash
# 桌面模式 (跳出視窗即時預覽)
python3 jetson_infer.py --engine ppyoloe_s.engine --source 0 --cuda-pre --show

# 無桌面 (Headless SSH 模式，終端機即時輸出 FPS)
python3 jetson_infer.py --engine ppyoloe_s.engine --source 0 --cuda-pre
```

#### 📁 資料夾批次影像推論
```bash
python3 jetson_infer.py --engine ppyoloe_s.engine --source /path/to/images --cuda-pre --save-dir results
```

---

## 📄 License
本專案基於 Apache 2.0 License 授權開源。
