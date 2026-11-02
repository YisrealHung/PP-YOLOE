# 🚀 PP-YOLOE-Lite-Train (含 NVIDIA Jetson Orin TensorRT 極速部署)

> 專為 **Google Colab (T4 GPU)** 與 **NVIDIA Jetson Orin NX / Nano** 深度優化之 PP-YOLOE+ 極速訓練與部署套件。  
> 專案開源於：**https://github.com/YisrealHung/PP-YOLOE**  
> 融合 **PP-YOLOE-R 旋轉框的精準架構設計** 與 **DAMO-YOLO-Lite-Train 記憶體 Byte-Cache 技術**，將原本 2~3 小時的訓練時間驟降至 **15~20 分鐘**！

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
| **50 輪耗時 (3,500張圖)** | 約 2 小時 15 分 | 約 50 ~ 65 分 | **約 15 ~ 18 分鐘 (提速 7~8 倍！)** |

---

## ⚡ PP-YOLOE-Lite-Train 核心加速策略

1. **標籤分配器全面減負 (`static_assigner_epoch: -1`)**：
   PP-YOLOE 原本設計在訓練初期透過 ATSS 暖機。但使用 Objects365 海量預訓練權重時，骨幹特徵早已具備強大表徵能力，ATSS 的暖機純屬浪費！本專案設為 `-1`，直接使用 `TaskAlignedAssigner`。
2. **流線型單一尺度資料管道 (Streamlined Fast Pipeline)**：
   移除 Python CPU 端耗時的 `RandomDistort` 與 `RandomExpand`，並移除多尺度動態縮放。保留 `PadGT` 以生成 `pad_gt_mask`，兼具最高速度與標籤分配精準度。
3. **RAM Byte-Cache 記憶體快取**：
   在 `ppdet.data.transform.operators.Decode` 注入記憶體快取，將壓縮後的影像 Byte 暫存在記憶體中（5,000 張 $1024 \times 1024$ 圖片僅佔約 770 MB RAM）。第 2 輪起完全免除硬碟讀取，終結 DataLoader 飢餓。
4. **硬體滿載 Batch Size 32 + AMP + cuDNN Benchmark**：
   在 T4 GPU 16GB 顯存下，Batch Size 32 僅佔用約 6GB 顯存。每輪僅需 109 次迭代即可完成（約 16 秒/輪），訓練步數縮減為 1/4！
5. **Jetson Orin NX 零 Plugin 直轉 ONNX**：
   導出 `exclude_nms=True` 靜態輸入維度 ONNX，在板端使用原廠 `trtexec` 一鍵直轉 TensorRT Engine，免編譯任何自訂 C++ Plugin！

---

## 📁 專案目錄結構

```text
PP-YOLOE-Lite-Train/
├── configs/
│   └── ppyoloe_plus_crn_s_lite.yml     # PP-YOLOE-Lite 極速訓練參考設定檔
├── tools_custom/
│   ├── patch_paddle_lite.py            # 自動注入 RAM Byte-Cache 與 NumPy 2.0 Shim
│   └── make_config.py                  # 動態自動設定檔生成器
├── deploy_jetson/
│   ├── build_engine.sh                 # Jetson Orin TensorRT FP16 一鍵編譯腳本
│   ├── ppyoloe_trt_utils.py            # 板端前處理、向量化後處理與 PyCUDA 推論類
│   └── jetson_infer.py                 # 板端即時推論主程式 (圖片/資料夾/攝影機串流/影片)
├── PP_YOLOE_Lite_Train.ipynb           # Google Colab 一鍵極速訓練互動式 Notebook
├── requirements.txt                    # 依賴套件清單
├── .gitignore                          # Git 忽略設定
└── README.md                           # 專案說明文件
```

---

## 🚀 快速上手：Google Colab 極速訓練

1. 將 `PP_YOLOE_Lite_Train.ipynb` 上傳至 Google Colab（或透過 GitHub 直接開啟）。
2. 在選單「執行階段 → 變更執行階段類型」選取 **T4 GPU**。
3. 在第 0 節「⚙️ 參數設定」中填寫你的 Google Drive 資料集路徑（支援 YOLO txt 或 COCO JSON zip）。
4. 點選「執行階段 → 全部執行」即可體驗 15 分鐘極速收斂！

---

## 🤖 NVIDIA Jetson Orin NX (TensorRT) 實機部署指南

訓練完成後，Colab 會在 Google Drive 的 `ppyoloe_runs/<專案名稱>/export/` 產生專屬部署包。

### 1. 將部署包傳輸至 Jetson Orin
```bash
scp -r ./export/* jetson@<JETSON_IP>:~/ppyoloe_deploy/
```

### 2. 開啟 Orin 最高效能模式
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 3. 一鍵編譯 TensorRT Engine (免 Plugin)
```bash
cd ~/ppyoloe_deploy
bash build_engine.sh
```

### 4. 執行即時串流推論
```bash
# USB / CSI 攝影機即時推論 (攝影機編號 0，顯示即時 FPS)
python3 jetson_infer.py --engine ppyoloe_plus_s_640.engine --source 0 --show

# 單張圖片或資料夾批次推論
python3 jetson_infer.py --engine ppyoloe_plus_s_640.engine --source test.jpg --save-dir results
```

