# Medicine Box AI Final Demo

本仓库当前目标：先跑通电脑端药盒检测 demo。Grove Vision AI V2 相关串口、刷写、AT 调用代码保留，但默认不依赖 Grove 实机。

## 当前可运行内容

- 电脑端独立 demo：生成多样化合成药盒图，训练轻量 detector，推理检测 `medicine_box`、`barcode`、`text`，输出 JSON、评估指标、YOLO 数据集和带框图片。
- 合成变化：旋转、多盒、遮挡、多背景、不同药盒配色、中文/英文/商标等小字。
- 单图 fallback：`--image path.jpg` 可以对已有图片跑同一 detector。
- Grove 辅助脚本：保留 `scripts/grovevision_at.py`、`scripts/deploy_we2_model.py`、`scripts/run_hybrid_demo.py`，但不是默认验收路径。
- 文档和报告：`reports/` 内保留架构、训练总结、数据计划和演讲稿。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 scripts/run_host_synthetic_demo.py --force-train
```

成功时终端会看到类似：

```text
PASS demo_000.jpg: medicine_box=1 barcode=1 text=3
PASS demo_001.jpg: medicine_box=1 barcode=1 text=3
```

默认输出：

```text
out/host_synthetic_demo/
  synthetic_detector_profile.json
  yolo_dataset/data.yaml
  test_fixed/manifest.json
  demo/images/*.jpg
  results/*.json
  results/*_overlay.jpg
  results/metrics.json
  results/summary.json
```

## 电脑端 Demo 做了什么

`scripts/run_host_synthetic_demo.py` 从零执行完整 host-side 流程：

1. 生成合成药盒训练/验证/固定测试/demo 图和标签。
2. 从生成标签校准轻量颜色/几何 detector。
3. 导出标准 YOLO 数据集。
4. 在固定测试集记录 `mAP50_proxy`、F1、漏检率和延迟。
5. 推理检测整盒药盒、条码区域和文字区域。
6. 写入结构化 JSON。
7. 生成带框 overlay 图片。

Overlay 示例：

![Host synthetic overlay example](reports/host_synthetic_overlay_example.jpg)

示例检测 JSON 包含：

```json
{
  "counts": {
    "medicine_box": 1,
    "barcode": 1,
    "text": 3
  },
  "detections": [
    {
      "label": "medicine_box",
      "score": 0.99,
      "bbox_xyxy": [228, 41, 749, 454],
      "bbox_xywh": [228, 41, 521, 413],
      "text_hint": "whole package 0"
    },
    {
      "label": "barcode",
      "score": 0.98,
      "bbox_xyxy": [507, 296, 676, 395],
      "bbox_xywh": [507, 296, 169, 99],
      "text_hint": "6908087349415"
    }
  ]
}
```

说明：当前脚本中的 barcode 数字和 text 内容来自合成图生成标签，用于 demo 验证，不等同真实 OCR 或真实条码解码。

## 单张图片 Fallback

```bash
python3 scripts/run_host_synthetic_demo.py \
  --image /path/to/image.jpg \
  --output out/single_image_check
```

这会输出：

```text
out/single_image_check/results/<image>.json
out/single_image_check/results/<image>_overlay.jpg
```

注意：fallback 使用合成图颜色/几何 detector。真实图片泛化能力取决于图片是否接近生成图风格。

## 评估与 YOLO 数据

默认 demo 会生成固定测试集并输出：

```text
out/host_synthetic_demo/results/metrics.json
out/host_synthetic_demo/yolo_dataset/data.yaml
```

也可以单独评估：

```bash
python3 scripts/evaluate_host_synthetic_detector.py \
  --manifest out/host_synthetic_demo/test_fixed/manifest.json \
  --model out/host_synthetic_demo/synthetic_detector_profile.json \
  --output out/host_synthetic_demo/results/eval_metrics.json
```

YOLOv8n 训练路线（可选，需要自行安装 `ultralytics`）：

```bash
pip install ultralytics
yolo detect train \
  model=yolov8n.pt \
  data=out/host_synthetic_demo/yolo_dataset/data.yaml \
  imgsz=640 \
  epochs=30 \
  project=out/yolo_runs \
  name=medicine_box_synth
```

真实微调路线：

1. 用同一类别表标注真实图片：`0 medicine_box`, `1 barcode`, `2 text`。
2. 保持 YOLO 目录结构：`images/train`, `images/val`, `labels/train`, `labels/val`。
3. 先用合成数据预训练，再用真实数据继续训练：

```bash
yolo detect train \
  model=out/yolo_runs/medicine_box_synth/weights/best.pt \
  data=/path/to/real_medicine_box_yolo/data.yaml \
  imgsz=640 \
  epochs=20 \
  project=out/yolo_runs \
  name=medicine_box_real_finetune
```

## Grove 端状态

Grove 端代码暂不作为默认运行要求：

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 probe
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 50 --out out/grove_check
```

刷写脚本帮助页不再强制要求 `pyserial`：

```bash
python3 scripts/deploy_we2_model.py --help
```

真正刷写模型时仍需要 `pyserial` 和 Grove 实机。

## 主要目录

```text
scripts/
  run_host_synthetic_demo.py  # 默认电脑端合成 demo
  run_hybrid_demo.py          # Grove 触发 + 主机 pipeline 串联脚本
  grovevision_at.py           # Grove AT 命令 helper
  deploy_we2_model.py         # Grove WE2 模型刷写 helper

local_train/
  generate_medicine_box_data.py
  medicine_box/data.yaml

reports/
  final_architecture.md
  local_training_summary.md
  dataset_needed.md
```

## 验证命令

```bash
python3 -m compileall -q scripts
python3 -m unittest discover -s tests -v
python3 scripts/deploy_we2_model.py --help
python3 scripts/run_host_synthetic_demo.py --force-train --demo-count 2
python3 scripts/evaluate_host_synthetic_detector.py
```

## 已知限制

- 默认 demo 针对生成图，保证可复现，不代表真实药盒照片泛化能力。
- 当前不要求 Grove 硬件接入，不验证串口 probe/invoke/flash。
- 合成 demo 的 text/barcode 字符串来自生成标签，不是真实 OCR/条码解码结果。
- 真实世界版本仍需采集真实药盒图、标注 whole-box/barcode/text，并训练更强 detector。
