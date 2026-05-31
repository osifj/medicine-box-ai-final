# Medicine Box AI — 5/30 进展 & Demo 就绪状态

## 今天完成

### 1. 混合 demo 脚本重写
`scripts/run_hybrid_demo.py` 已完全重写，修复了原来的 bug 并加了三项关键功能：

- `--tscore 25`：自动发 AT+TSCORE 降 Grove 检测阈值（原默认 45，你之前测不到就是因为这个）
- `--wait`：循环轮询模式，每秒 invoke 等药盒出现，不用手动算时机
- 美化输出：Pipeline 跑完直接显示药名、置信度、OCR 文字

原来有两个参数 bug：`--capture-once`不存在（应为`--max-frames 1`），`--output-prefix`不存在（应为`--save-dir`或`--output`），均已修正。

### 2. 电脑端 pipeline 全量评估

**Detector 指标**（fintetune_v4, 真实 val 14 图）：
| 类 | P | R | F1 | GT |
|---|---|---|---|---|
| text | 0.744 | 0.762 | 0.753 | 42 |
| barcode | N/A | N/A | N/A | 0（val 中无 barcode GT） |

**E2E 识别**（detect → OCR → classify，14 张 val 图）：
- 12/14 正确识别（86%），平均 1.5s/张
- 识别出：布洛芬缓释胶囊、阿莫西林胶囊
- 2 张未识别（frame_00003, frame_00004，detector 检出太少）

主要 FP 来源：高置信度窄条（aspect>15），detector 对横边过敏感。
主要 FN 来源：超小文字 badge 面积<0.005，26×26 grid 难以捕捉。

## Demo 当天操作

Grove 插上后：

```bash
cd /Users/dep/Documents/Codex/2026-05-28/gorvevision-grovevision-at-command-arduino-grove

# 完整链路（推荐）
python3 scripts/run_hybrid_demo.py --wait --tscore 25

# 只测 Grove 端（确认检测正常）
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 10 --out out/demo_test

# 只测电脑端（已有图片）
cd /Users/dep/Projects/New_project
python3 scripts/run_image_pipeline.py --image captures/host_session_001/images/frame_00000.jpg --enable-detector
```

## 已知局限

- barcode 解码不稳定（检出区域但 zbar 未能解码）
- OCR 对中文识别差（Tesseract，小字中文基本乱码），靠英文药名匹配兜底
- Grove 240×240 分辨率，药盒需靠近（10-20cm）且正面朝镜头

## 项目路径

- Grove 端：`/Users/dep/Documents/Codex/2026-05-28/gorvevision-grovevision-at-command-arduino-grove`
- 电脑端：`/Users/dep/Projects/New_project`
- Grove 串口：`/dev/cu.usbmodem5B420573151`，921600 baud
