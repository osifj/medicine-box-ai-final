# Local Medicine Box Model Training - Summary Report

Date: 2026-05-28

## 1. Can we skip SenseCraft web training entirely?

**Partially.** Local YOLOv8 training works: YOLOv8n trained on synthetic data achieves mAP50=0.978. But three blockers prevent YOLOv8 deployment:

- **Int8 quantization**: TF Lite converter crashes on YOLOv8 CONCATENATION ops
- **Ethos-U NPU**: Vela reports 0% NPU — float32 feature maps not supported
- **Output format**: YOLOv8 outputs raw [1,5,756]; Grove expects decoded [x,y,w,h,score,target]

## 2. Did YOLOv8 succeed for training?

**Yes.** YOLOv8n, 64 synthetic images at 192x192, 100 epochs on Apple M1. Best mAP50=0.978. ONNX export (11.6 MB) works.

## 3. Is SSCMA / Swift-YOLO the right path?

**Yes.** Seeed ModelAssistant cloned at local_train/sscma_model_assistant/. Contains RTMDet configs for MCU deployment. Requires mmengine/mmdet install.

## 4. Was medicine_box_int8_vela.tflite generated?

**No.** Int8 export failed at the TF converter level. Float32 and dynamic-range TFLite files exist but are not Grove-compatible (0% Ethos-U NPU usage).

## 5. Did Python flashing succeed?

**YES — major breakthrough.** Fixed 6 bugs in deploy_we2_model.py:
1. pyserial rts/dtr must be set after serial.Serial(), not as constructor args
2. XMODEM block 0 is valid after block 255 (removed incorrect 0->1 reset)
3. Post-config: need '1' trigger, not flash_complete
4. Post-data: device shows [0] Reboot / [1] Xmodem menu, not yes/no prompt
5. EOT: device skips standard XMODEM EOT ACK (simplified to fire-and-forget)
6. Metadata: AT+INFO write after reboot

Flashing time: ~20 min for 2.8 MB model (22256 blocks x 20ms delay).

## 6. Did invoke return boxes?

**Yes — invoke works.** Returns valid JSON with JPEG image. Boxes are empty because the model is "Storage box Detection" (ID 60261), not trained for medicine boxes. The camera needs a real medicine box in view.

Device status: Model: Storage box Detection, Classes: 0=Storage box

## 7. What failed?

| Step | Status |
|------|--------|
| Real data collection | Not done (synthetic only) |
| YOLOv8 training | Works (mAP50 0.978) |
| Int8 quantization | TF concat bug |
| Vela compilation | 0% NPU (float32) |
| YOLOv8 output format | Raw tensor, not SSCMA boxes |
| Python flash script | FIXED AND WORKING |
| SenseCraft model flash | Storage Box model flashed |
| Grove invoke | Valid JSON, boxes empty |

## 8. Tomorrow

### Fastest path to working medicine_box model:
1. Re-annotate 96 existing images with whole-box medicine_box labels
2. Use SenseCraft AI web to train Swift-YOLO on the annotated dataset
3. Download the trained model and flash with our working deploy script
4. Verify: probe shows "Medicine Box Detection", invoke returns boxes

### Or full local SSCMA:
1. Create venv, install SSCMA/ModelAssistant with mmengine/mmdet
2. Train RTMDet nano, 1 class (medicine_box), 192x192 input
3. Export to SSCMA TFLite format, Vela compile, flash

### Test with real box:
Place a medicine box in front of Grove camera and run:
  python3 scripts/grovevision_at.py --port /dev/cu.usbmodem5B420573151 invoke --image --min-score 20 --out out

### Flash script improvements:
- Reduce inter-block delay to 5-10ms
- Auto-detect and retry failed metadata writes
- Add progress bar
