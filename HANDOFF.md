# Grove Vision AI V2 药盒检测项目交接说明

## 当前目标

让 Grove Vision AI V2 识别药盒区域，并返回/显示目标框。用户希望通过 AT Command / Arduino 控制 Grove Vision。

## 当前工作目录

```text
/Users/dep/Documents/Codex/2026-05-28/gorvevision-grovevision-at-command-arduino-grove
```

## 当前硬件状态

电脑已经识别到 Grove Vision AI V2 串口：

```text
/dev/cu.usbmodem5B420573151
```

USB 设备信息：

```text
Vendor ID: 0x1a86
Product ID: 0x55d3
Serial Number: 5B42057315
```

AT 串口参数：

```text
921600 baud, 8N1
```

注意：之前 Chrome / SenseCraft WebSerial 页面曾占用串口。排查命令：

```bash
lsof /dev/cu.usbmodem5B420573151
```

如果有 Chrome、Arduino IDE、串口监视器占用，先关闭/Disconnect。

## 已确认的设备信息

用本地脚本 `scripts/grovevision_at.py` 已成功连接设备并读取：

```text
Device name: Grove Vision AI V2
Device ID: 98b62546
Firmware software: 2025.01.02
AT API: v0
Sensor: 240x240 Auto
```

当前已部署模型不是药盒模型，而是：

```text
Model: Gesture Detection
Classes:
0 = Paper
1 = Rock
2 = Scissors
```

因此当前对药盒推理返回：

```json
"boxes": []
```

这不是 AT 通信问题，而是模型问题。

## 已创建文件

```text
README.md
HANDOFF.md
scripts/grovevision_at.py
scripts/deploy_we2_model.py
arduino/GroveVisionMedicineBoxDetector/GroveVisionMedicineBoxDetector.ino
out/grovevision_frame.jpg
out/grovevision_result.html
```

### 1. AT 测试脚本

路径：

```text
scripts/grovevision_at.py
```

用途：

- 查询设备信息
- 查询当前模型信息
- 调用 `AT+INVOKE`
- 解析目标框
- 可选请求 JPEG 图像，并生成带框 HTML

已知必须使用 `pyserial` 打开串口。纯 `termios/stty` 方式之前无回包，但 pyserial 正常。

常用命令：

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 probe
```

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --min-score 50
```

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 50 --out out
```

输出图片：

```text
out/grovevision_frame.jpg
out/grovevision_result.html
```

当前抓到的图像是绿色偏色的低分辨率 Grove 相机帧，推理无框。

### 2. Arduino 示例

路径：

```text
arduino/GroveVisionMedicineBoxDetector/GroveVisionMedicineBoxDetector.ino
```

用途：

- 使用 `Seeed_Arduino_SSCMA` 库
- 通过 UART 调用 `AI.invoke(1, false, false)`
- 打印检测框：target、score、center、size、rect

注意：

- Arduino IDE 需要安装 `Seeed_Arduino_SSCMA`
- 普通 UNO 的 SoftwareSerial 不适合 `921600`
- 推荐 ESP32 / XIAO / 有硬件 `Serial1` 的板子
- 目前 `MEDICINE_BOX_TARGET = -1`，表示打印所有框；药盒模型部署后改成药盒 class id

### 3. WE2 模型部署脚本

路径：

```text
scripts/deploy_we2_model.py
```

用途：

- 复刻 Seeed SenseCraft Web Toolkit 的 Grove AI WE2 刷写流程
- 进入 Himax bootloader
- 用 XMODEM 写 offset config
- 把 `.tflite` 写到 `0x400000`
- 重启后用 `AT+INFO` 写模型 metadata

本脚本是为已有 Grove Vision AI V2 可用模型准备的，模型文件通常应是：

```text
*_int8_vela.tflite
```

真正部署本地药盒模型命令示例：

```bash
python3 scripts/deploy_we2_model.py \
  --port /dev/cu.usbmodem5B420573151 \
  --model-file /absolute/path/to/medicine_box_int8_vela.tflite \
  --classes medicine_box \
  --name "Medicine Box Detection"
```

只下载公开 SenseCraft 模型检查，不刷写：

```bash
python3 scripts/deploy_we2_model.py \
  --sensecraft-model-id 60261 \
  --download-only \
  --metadata-out models/storage_box_metadata.json
```

## 关键技术结论

### AT 协议

核心命令：

```text
AT+ID?
AT+NAME?
AT+STAT?
AT+VER?
AT+MODEL?
AT+INFO?
AT+SENSOR?
AT+INVOKE=1,0,1
AT+INVOKE=1,0,0
```

`AT+INVOKE=1,0,1`：

- 推理 1 次
- 不做 differed 输出
- 只返回结果

`AT+INVOKE=1,0,0`：

- 推理 1 次
- 返回结果和 JPEG 图像

Grove 返回框格式：

```text
[x, y, w, h, score, target]
```

其中 `x/y` 是中心点，不是左上角。左上角：

```text
left = x - w / 2
top = y - h / 2
```

### 当前最大阻塞

设备通信、图像获取、推理调用都已跑通。

阻塞点是：板子上没有药盒目标检测模型。

当前模型是手势检测，所以无法框药盒。

## 关于已有本机模型

用户机器上发现了旧项目：

```text
/Users/dep/Projects/New_project
/Users/dep/Documents/New project
```

里面有多个 Keras 模型：

```text
artifacts/detector_synth_v2/detector_best.keras
artifacts/detector_synth_v3/detector_best.keras
artifacts/detector_synth_v3b/detector_best.keras
artifacts/detector_finetune_v4/detector_best.keras
...
```

对应 metadata 显示类别主要是：

```text
barcode
text
```

这些模型用于药盒上的条形码/文字区域检测，不是整盒 `medicine_box` 检测。

更重要的是，这些 Keras 模型是自定义 TensorFlow 检测头，不是 Seeed SSCMA / Swift-YOLO 模型格式。即使导出成普通 int8 TFLite，也很可能不能被 Grove Vision AI V2 固件正确解析为 `boxes`。

不要直接把这些 Keras/TFLite 刷进 Grove，除非已经确认 SSCMA-Micro 支持该输出结构。

## 推荐下一步

### 路线 A：最快可演示，刷一个“盒子检测”近似模型

公开 SenseCraft 模型中找到：

```text
ID 60261: Storage box Detection
ID 60398: Computer Box Detection
ID 60269: Pillow Detection
```

其中 `Storage box Detection` 可能可作为临时“盒子检测”近似，但它不是真正药盒模型。

下载检查：

```bash
python3 scripts/deploy_we2_model.py \
  --sensecraft-model-id 60261 \
  --download-only \
  --metadata-out models/storage_box_metadata.json
```

如用户接受“临时近似”，再去掉 `--download-only` 刷写：

```bash
python3 scripts/deploy_we2_model.py \
  --port /dev/cu.usbmodem5B420573151 \
  --sensecraft-model-id 60261 \
  --metadata-out models/storage_box_metadata.json
```

刷写后验证：

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 probe
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 50 --out out
```

### 路线 B：正确项目路线，训练真正 `medicine_box` 模型

使用 SenseCraft AI / Grove Vision AI V2 workspace：

1. 打开 SenseCraft AI
2. 选择 Grove Vision AI V2 / Grove AI WE2
3. 用相机采集药盒图像
4. 标注单类 `medicine_box`
5. 训练 Object Detection / Swift-YOLO 模型
6. 导出/部署到 Grove Vision AI V2
7. 验证 `AT+INFO?` classes 中出现 `medicine_box`
8. 再运行 `invoke --image`

注意：之前无法自动操作 Chrome，因为当前 Chrome 没有安装/启用 Codex Chrome Extension。用户需要自己在网页操作，或安装扩展后再让代理继续。

### 路线 C：本地模型路线

如果 DeepSeek 能训练/导出 Grove Vision AI V2 兼容模型，目标文件应是：

```text
medicine_box_int8_vela.tflite
```

然后用：

```bash
python3 scripts/deploy_we2_model.py \
  --port /dev/cu.usbmodem5B420573151 \
  --model-file /absolute/path/to/medicine_box_int8_vela.tflite \
  --classes medicine_box \
  --name "Medicine Box Detection"
```

模型必须符合 SSCMA-Micro 固件的目标检测输出格式，否则即使刷入也不会返回正确 `boxes`。

## 验证成功标准

部署成功后：

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 probe
```

应看到：

```text
Model: Medicine Box Detection
Classes: 0=medicine_box
```

推理：

```bash
python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 50 --out out
```

应看到类似：

```text
Detected boxes:
  [0] label=medicine_box target=0 score=...
```

并生成：

```text
out/grovevision_frame.jpg
out/grovevision_result.html
```

## 参考来源

- Seeed Grove Vision AI V2 AT 文档
  - https://wiki.seeedstudio.com/grove_vision_ai_v2_at/
- Seeed Arduino SSCMA 库
  - https://github.com/Seeed-Studio/Seeed_Arduino_SSCMA
- SenseCraft Web Toolkit 刷写流程参考
  - https://github.com/Seeed-Studio/SenseCraft-Web-Toolkit
- SenseCraft Model Assistant
  - https://github.com/Seeed-Studio/ModelAssistant

