# Medicine Box AI Final Demo

本仓库当前目标：先跑通电脑端药盒检测 demo。Grove Vision AI V2 相关串口、刷写、AT 调用代码保留，但默认不依赖 Grove 实机。

## 当前可运行内容

- 电脑端独立 demo：生成合成药盒图，训练轻量 detector，推理检测 `medicine_box`、`barcode`、`text`，输出 JSON 和带框图片。
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
  demo/images/*.jpg
  results/*.json
  results/*_overlay.jpg
  results/summary.json
```

## 电脑端 Demo 做了什么

`scripts/run_host_synthetic_demo.py` 从零执行完整 host-side 流程：

1. 生成合成药盒训练图和标签。
2. 从生成标签校准轻量颜色/几何 detector。
3. 生成 demo 输入图。
4. 推理检测整盒药盒、条码区域和文字区域。
5. 写入结构化 JSON。
6. 生成带框 overlay 图片。

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
      "bbox_xyxy": [138, 71, 794, 411]
    }
  ]
}
```

说明：当前脚本中的 barcode 数字和 text 内容来自合成图生成标签，用于 demo 验证，不等同真实 OCR 或真实条码解码。

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
```

## 已知限制

- 默认 demo 针对生成图，保证可复现，不代表真实药盒照片泛化能力。
- 当前不要求 Grove 硬件接入，不验证串口 probe/invoke/flash。
- 合成 demo 的 text/barcode 字符串来自生成标签，不是真实 OCR/条码解码结果。
- 真实世界版本仍需采集真实药盒图、标注 whole-box/barcode/text，并训练更强 detector。
