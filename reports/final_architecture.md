# Final System Architecture

## 系统总览

本文档描述药盒检测与识别系统的最终架构。系统由两大部分组成：

- **Grove Vision AI V2（边缘端）**：低功耗、低分辨率，只负责整盒药盒检测
- **电脑端（主机端）**：高清摄像头 + barcode/OCR，负责精确识别

```
┌─────────────────────────────────────────────────────────┐
│                    用户放入药盒                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│           Grove Vision AI V2（边缘触发端）                │
│                                                         │
│  摄像头: 240×240 板载摄像头                               │
│  模型: medicine_box 单类目标检测                          │
│  输出: [x, y, w, h, score, target]                      │
│                                                         │
│  职责:                                                   │
│  ✅ 检测画面中是否有药盒                                   │
│  ✅ 返回药盒外框坐标                                       │
│  ❌ 不识别文字                                            │
│  ❌ 不解码barcode                                        │
│  ❌ 不做OCR                                              │
└─────────────────────┬───────────────────────────────────┘
                      │ 检测到 medicine_box？
                      │
              ┌───────┴───────┐
              │ YES           │ NO
              ▼               ▼
    触发主机拍照          提示用户调整药盒位置
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              电脑端（主识别路径）                          │
│                                                         │
│  摄像头: Mac/USB 高清摄像头                                │
│  模型: barcode/text 双类检测器 + OCR 模型                  │
│                                                         │
│  职责:                                                   │
│  ✅ 拍摄高清药盒图像                                       │
│  ✅ 检测条形码/二维码区域                                   │
│  ✅ 检测文字区域                                          │
│  ✅ OCR 识别药盒上的药品名、批号等信息                       │
│  ✅ 结构化输出 JSON                                       │
│  ✅ 生成可视化 preview 图                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   最终输出                                │
│                                                         │
│  {                                                      │
│    "grove": {"detected": true, "boxes": [...]},         │
│    "host_camera": {"barcode": [...], "ocr": [...]},     │
│    "medicine_info": {...},                               │
│    "final_status": "success"                             │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

## Grove Vision AI V2 角色

### 硬件定位
- 边缘端低功耗 AI 推理
- 板载 240×240 摄像头
- Ethos-U55 NPU 加速
- 项目硬件展示端

### 模型要求
- **单类**：`medicine_box`（class id = 0）
- **输入**：240×240（或 192×192）RGB
- **输出**：`[x_center, y_center, width, height, score, target]`
- **格式**：int8 + Vela 优化的 TFLite（SSCMA 兼容）

### 明确不做的
- ❌ 不识别药盒上的小字
- ❌ 不解码条形码
- ❌ 不做 OCR
- ❌ 不替代电脑端识别流程
- ❌ 不跑 barcode/text 双类检测器

### 为什么 Grove 只做整盒检测
- Grove 摄像头分辨率（240×240）不足以清晰拍摄 barcode 细线和中文小字
- 但足以区分"画面里有没有药盒"——这是一个粗粒度任务
- 作为边缘 AI 硬件 demo，展示低功耗端侧推理
- 作为触发器，省去电脑端持续推理的功耗

## 电脑端角色

### 硬件定位
- Mac / USB 高清摄像头（通常 1920×1080 或更高）
- CPU/GPU 推理（TensorFlow）

### 软件组件
- **detector_model.py**：barcode + text 双类检测器（已有）
- **ocr_model.py**：文字识别模型（已有）
- **pipeline.py**：端到端识别 pipeline（已有）
- **medicine_classifier.py**：药品种类分类（已有）

### 职责
- 拍摄高清药盒图像
- 检测条形码区域并解码
- 检测文字区域并 OCR
- 提取药品名、批号、规格等信息
- 输出结构化 JSON
- 生成可视化结果图

## 为什么不能用 Grove 替代电脑端

| 维度 | Grove Vision AI V2 | 电脑高清摄像头 |
|------|-------------------|---------------|
| 分辨率 | 240×240 | 1920×1080+ |
| 条形码细线 | 不可分辨 | 清晰可辨 |
| 中文小字 | 模糊 | 锐利 |
| OCR 精度 | 不可用 | 高 |
| 定位 | 边缘触发 | 主识别 |
| 功耗 | 低（<0.3W） | 高 |

## 双摄像头坐标问题

Grove 摄像头和电脑摄像头视角不同，**不能直接映射坐标**。

- Grove 的 box 坐标只证明"Grove 看到了药盒"
- 电脑端必须用自己的 detector 在高清图上重新检测 barcode/text 区域
- 除非做双摄标定（复杂且不必要），否则不做 Grove → 电脑坐标系映射

## 最终 Demo 流程

1. 用户将药盒放在 Grove 摄像头前
2. `run_hybrid_demo.py` 调用 Grove probe + invoke
3. Grove 检测到 `medicine_box` → 触发
4. 电脑端拍摄高清图像
5. 电脑端运行 barcode + OCR pipeline
6. 输出统一 JSON 和预览图

## 项目目录

```
Grove 端:
  /Users/dep/Documents/Codex/2026-05-28/gorvevision-grovevision-at-command-arduino-grove/
  ├── scripts/grovevision_at.py          # AT 命令测试脚本
  ├── scripts/deploy_we2_model.py        # Python 刷写脚本（已修复）
  ├── local_train/                       # 本地训练目录
  ├── out/                               # 推理输出
  └── reports/                           # 文档

电脑端:
  /Users/dep/Projects/New_project/
  ├── medicine_box_vision/               # 核心库
  │   ├── detector_model.py              # barcode/text 检测器
  │   ├── detector_data.py               # 数据 pipeline
  │   ├── detector_train.py              # 训练入口
  │   ├── ocr_model.py                   # OCR 模型
  │   ├── pipeline.py                    # 端到端 pipeline
  │   ├── box_locator.py                 # 传统视觉整盒定位
  │   └── export.py                      # TFLite 导出
  ├── scripts/
  │   ├── generate_synthetic_dataset.py  # 合成数据生成
  │   ├── run_host_camera.py             # 主机摄像头推理
  │   └── run_image_pipeline.py          # 图片 pipeline
  ├── data/
  ├── configs/
  └── artifacts/
```

## 成功标准

### Grove 端
```
$ python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 probe
Model: Medicine Box Detection
Classes: 0=medicine_box

$ python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 50 --out out
返回 boxes != []
label=medicine_box
```

### 电脑端
```
$ cd /Users/dep/Projects/New_project
$ python3 scripts/run_image_pipeline.py --image <path>
返回 barcode + OCR 结果 JSON
生成 preview 图
```

### Hybrid Demo
```
$ python3 scripts/run_hybrid_demo.py
Grove 检测 → 触发主机拍照 → barcode + OCR → 统一 JSON 输出
```
