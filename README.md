# Smart Room Presence Detection

Deployment snapshot of a university smart-room prototype: an int8 thermal occupancy model and an Arduino UNO R4 WiFi sketch using an MLX90640 24×32 thermal camera.

## Included

- `models/presence_model_int8.tflite`: 4,272-byte quantized model.
- `models/arduino_presence_model/model_int8.h`: the same model embedded as a C byte array.
- `models/arduino_presence_model/arduino_presence_model.ino`: sensor, inference, relay/LED and local Wi-Fi server integration.
- `models/deployment_report.json`: historical model/pruning evaluation metadata; developer-specific absolute paths have been replaced by placeholders.

This archive does **not** include the training pipeline, raw thermal/RGB dataset, course papers, student reports, presentation slides, or Wi-Fi credentials. It is not a fully reproducible training repository. Metrics in the historical report have not been independently reproduced in this publication.

## Hardware and libraries

The sketch targets Arduino UNO R4 WiFi and references MLX90640, AHTX0 and ADXL345 sensors. Install the UNO R4 board support, Adafruit sensor libraries and a compatible TensorFlow Lite Micro library separately through your preferred Arduino setup. This repository does not redistribute those libraries or guarantee compatibility with every library version.

Before building, copy `arduino_secrets.example.h` to `arduino_secrets.h` inside the sketch directory and set your own network credentials. The real file is excluded from Git. Select **Arduino UNO R4 WiFi** as the target board.

The sketch exposes a local HTTP server without a production authentication layer. Only test on a trusted, isolated network; do not expose it directly to the Internet. Review relay wiring and electrical safety before connecting equipment.

## Model inputs

The model expects a 24×32 thermal frame, normalized with training mean 27.0598087310791 and standard deviation 1.093957543373108, followed by the quantization parameters stored in the model. The deployed probability threshold is 0.541930. See the sketch and deployment report for implementation details.

## Data provenance

The original working directory used the public [FIR Image Action Localisation Dataset](https://github.com/visiongo-kr/FIR-Image-Action-Localisation-Dataset), whose README also points to the [earlier FIR Image Action Dataset](https://github.com/visiongo-kr/FIR-Image-Action-Dataset). The raw data is deliberately not mirrored here. Consult the original authors for dataset access, terms and citation requirements. No additional license is asserted over third-party data or dependencies.

## Verification

For this archive, the embedded byte array was checked against the `.tflite` file, and the report's model size was checked for consistency. No hardware build or live sensor test was performed during publication. The training script and historical build workspace are not included.

## 中文说明

这是 510 智能房间项目的精简部署归档，仅包含 Arduino 代码、量化模型和历史部署报告；不公开原始数据集、课程报告、演示文稿或 Wi-Fi 密码。它不是完整训练代码库，本次发布未重新训练或验证硬件。

使用前请自行安装依赖，将示例凭据文件复制为 `arduino_secrets.h` 后填写自己的网络信息。仅在可信隔离网络测试，不要将无生产级认证的 HTTP 服务暴露到公网。
