#include <Wire.h>

#if !defined(ARDUINO_UNOR4_WIFI)
#error "This sketch targets Arduino UNO R4 WiFi. In Arduino IDE select Tools > Board > Arduino UNO R4 Boards > Arduino UNO R4 WiFi."
#endif

#include <WiFiS3.h>
#include <Adafruit_MLX90640.h>
#include <Adafruit_AHTX0.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_int8.h"

// --- WIFI CREDENTIALS ---
#include "arduino_secrets.h"
const char ssid[] = SECRET_SSID;
const char pass[] = SECRET_PASS;

WiFiServer server(80);

// --- SENSOR OBJECTS ---
Adafruit_MLX90640* mlx = nullptr;
Adafruit_AHTX0 aht;
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
float* frame = nullptr;

// --- MODEL NORMALIZATION / DECISION ---
constexpr float TRAIN_MEAN_C = 27.0598087310791f;
constexpr float TRAIN_STD_C = 1.093957543373108f;
constexpr float MODEL_THRESHOLD = 0.541930f;

// The UNO R4 WiFi has only 32KB SRAM. Keep the arena small and allocate the
// thermal camera buffers on the heap so the sketch can still link.
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// --- WARM-PIXEL DEBUG BASELINE ---
const float MIN_HUMAN_TEMP = 30.0f;
const float MAX_HUMAN_TEMP = 40.0f;
const int MIN_PIXELS_NEEDED = 5;

// --- PIN DEFINITIONS ---
const int PIN_LED_RED = 2;
const int PIN_LED_YEL = 3;
const int PIN_LED_GRN = 4;
const int PIN_RELAY_1 = 5;
const int PIN_RELAY_2 = 6;

// --- STATE TRACKING VARIABLES ---
bool lastHumanState = false;
unsigned long absenceStartTime = 0;
bool isLeavingPhase = false;
bool piRelayOverride = false;

float modelScore = 0.0f;
bool modelDetected = false;
bool thresholdDetected = false;
int warmPixelCount = 0;
unsigned long inferenceMs = 0;
bool modelReady = false;

static float sigmoidf_safe(float x) {
  if (x >= 0.0f) {
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  float z = expf(x);
  return z / (1.0f + z);
}

static int tensorElementCount(const TfLiteTensor* tensor) {
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    count *= tensor->dims->data[i];
  }
  return count;
}

bool setupModel() {
  model = tflite::GetModel(presence_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(model->version());
    Serial.print(" != supported ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddReshape();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  if (model_input->type != kTfLiteInt8 || model_output->type != kTfLiteInt8) {
    Serial.println("Expected int8 input and output tensors");
    return false;
  }

  if (tensorElementCount(model_input) != 24 * 32) {
    Serial.println("Unexpected input size");
    return false;
  }

  return true;
}

bool runPresenceInference() {
  if (!modelReady) return false;

  const float inputScale = model_input->params.scale;
  const int inputZeroPoint = model_input->params.zero_point;
  int8_t* inputBuffer = model_input->data.int8;

  for (int i = 0; i < 24 * 32; ++i) {
    float normalized = (frame[i] - TRAIN_MEAN_C) / TRAIN_STD_C;
    int quantized = (int)lroundf(normalized / inputScale) + inputZeroPoint;
    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;
    inputBuffer[i] = (int8_t)quantized;
  }

  unsigned long start = millis();
  TfLiteStatus invokeStatus = interpreter->Invoke();
  inferenceMs = millis() - start;

  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke() failed");
    modelScore = 0.0f;
    modelDetected = false;
    return false;
  }

  int8_t rawOutput = model_output->data.int8[0];
  float logit = ((int)rawOutput - model_output->params.zero_point) * model_output->params.scale;
  modelScore = sigmoidf_safe(logit);
  modelDetected = modelScore >= MODEL_THRESHOLD;
  return true;
}

void updateRoomState(bool humanDetected) {
  if (humanDetected) {
    isLeavingPhase = false;
    lastHumanState = true;

    digitalWrite(PIN_LED_RED, HIGH);
    digitalWrite(PIN_LED_YEL, LOW);
    digitalWrite(PIN_LED_GRN, LOW);
    digitalWrite(PIN_RELAY_1, LOW);
    digitalWrite(PIN_RELAY_2, LOW);
  } else {
    if (lastHumanState) {
      lastHumanState = false;
      isLeavingPhase = true;
      absenceStartTime = millis();
    }

    if (isLeavingPhase) {
      if (millis() - absenceStartTime < 5000) {
        digitalWrite(PIN_LED_RED, LOW);
        digitalWrite(PIN_LED_YEL, HIGH);
        digitalWrite(PIN_LED_GRN, LOW);
        digitalWrite(PIN_RELAY_1, LOW);
        digitalWrite(PIN_RELAY_2, LOW);
      } else {
        isLeavingPhase = false;
      }
    }

    if (!isLeavingPhase) {
      digitalWrite(PIN_LED_RED, LOW);

      if (piRelayOverride) {
        digitalWrite(PIN_LED_YEL, HIGH);
        digitalWrite(PIN_LED_GRN, LOW);
        digitalWrite(PIN_RELAY_1, LOW);
        digitalWrite(PIN_RELAY_2, LOW);
      } else {
        digitalWrite(PIN_LED_YEL, LOW);
        digitalWrite(PIN_LED_GRN, HIGH);
        digitalWrite(PIN_RELAY_1, HIGH);
        digitalWrite(PIN_RELAY_2, HIGH);
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_LED_YEL, OUTPUT);
  pinMode(PIN_LED_GRN, OUTPUT);

  digitalWrite(PIN_RELAY_1, HIGH);
  digitalWrite(PIN_RELAY_2, HIGH);
  pinMode(PIN_RELAY_1, OUTPUT);
  pinMode(PIN_RELAY_2, OUTPUT);
  digitalWrite(PIN_LED_GRN, HIGH);

  Serial.println("Initializing TensorFlow Lite Micro model...");
  modelReady = setupModel();
  Serial.println(modelReady ? "Model initialized." : "Model initialization failed.");

  Wire.begin();
  Serial.println("Initializing sensors...");

  mlx = new Adafruit_MLX90640();
  frame = (float*)malloc(sizeof(float) * 32 * 24);
  if (mlx == nullptr || frame == nullptr) {
    Serial.println("Failed to allocate MLX90640 buffers!");
    while (1) delay(10);
  }

  if (!mlx->begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("Failed to find MLX90640!");
    while (1) delay(10);
  }
  mlx->setMode(MLX90640_CHESS);
  mlx->setResolution(MLX90640_ADC_18BIT);
  mlx->setRefreshRate(MLX90640_2_HZ);

  if (!aht.begin()) {
    Serial.println("Failed to find AHT20!");
    while (1) delay(10);
  }

  if (!accel.begin()) {
    Serial.println("Failed to find ADXL345!");
    while (1) delay(10);
  }

  Serial.println("Sensors initialized. Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    WiFi.begin(ssid, pass);
    delay(5000);
  }

  Serial.println("\nConnected to WiFi!");
  Serial.print("Arduino IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("HTTP Server started. Monitoring with ML model and waiting for Pi...");
  Serial.println("===============================================================");
}

void loop() {
  bool frameOk = false;
  warmPixelCount = 0;
  thresholdDetected = false;

  if (mlx->getFrame(frame) == 0) {
    frameOk = true;
    for (int i = 0; i < 768; i++) {
      if (frame[i] >= MIN_HUMAN_TEMP && frame[i] <= MAX_HUMAN_TEMP) {
        warmPixelCount++;
      }
    }
    thresholdDetected = warmPixelCount >= MIN_PIXELS_NEEDED;

    if (!runPresenceInference()) {
      modelDetected = false;
    }
  } else {
    Serial.print("[IR ERROR] ");
    modelDetected = false;
  }

  sensors_event_t humidity, temp;
  aht.getEvent(&humidity, &temp);

  sensors_event_t event;
  accel.getEvent(&event);

  bool humanDetected = modelReady ? modelDetected : thresholdDetected;
  updateRoomState(humanDetected);

  if (humanDetected && modelReady) Serial.print("[MODEL HUMAN] ");
  else if (humanDetected) Serial.print("[THRESHOLD HUMAN] ");
  else if (isLeavingPhase) Serial.print("[LEAVING] ");
  else Serial.print("[MODEL EMPTY] ");

  Serial.print("| model_score: "); Serial.print(modelScore, 4);
  Serial.print(" | model_detected: "); Serial.print(modelDetected ? "true" : "false");
  Serial.print(" | threshold_detected: "); Serial.print(thresholdDetected ? "true" : "false");
  Serial.print(" | warm_pixels: "); Serial.print(warmPixelCount);
  Serial.print(" | inference_ms: "); Serial.print(inferenceMs);
  Serial.print(" | Temp: "); Serial.print(temp.temperature, 1);
  Serial.print("C | Hum: "); Serial.print(humidity.relative_humidity, 1);
  Serial.print("% | Relays: ");
  Serial.println(digitalRead(PIN_RELAY_1) == LOW ? "ON" : "OFF");

  WiFiClient client = server.available();

  if (client) {
    while (client.connected() && !client.available()) {
      delay(1);
    }

    String request = client.readStringUntil('\r');
    client.flush();

    if (request.indexOf("GET /relay/on") != -1) {
      piRelayOverride = true;
      Serial.println(">>> PI COMMAND: Relay Override ON <<<");
    } else if (request.indexOf("GET /relay/off") != -1) {
      piRelayOverride = false;
      Serial.println(">>> PI COMMAND: Relay Override OFF <<<");
    } else {
      Serial.println(">>> Sending Data to Pi <<<");
    }

    bool isRelayOn = (digitalRead(PIN_RELAY_1) == LOW);

    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: application/json");
    client.println("Connection: close");
    client.println();

    client.print("{");
    client.print("\"model_score\":"); client.print(modelScore, 5); client.print(",");
    client.print("\"model_detected\":"); client.print(modelDetected ? "true" : "false"); client.print(",");
    client.print("\"threshold_detected\":"); client.print(thresholdDetected ? "true" : "false"); client.print(",");
    client.print("\"inference_ms\":"); client.print(inferenceMs); client.print(",");
    client.print("\"temperature_c\":"); client.print(temp.temperature); client.print(",");
    client.print("\"humidity_percent\":"); client.print(humidity.relative_humidity); client.print(",");
    client.print("\"relay_state\":"); client.print(isRelayOn ? "true" : "false"); client.print(",");
    client.print("\"frame_ok\":"); client.print(frameOk ? "true" : "false"); client.print(",");
    client.print("\"warm_pixel_count\":"); client.print(warmPixelCount); client.print(",");
    client.print("\"accel_x\":"); client.print(event.acceleration.x); client.print(",");
    client.print("\"accel_y\":"); client.print(event.acceleration.y); client.print(",");
    client.print("\"accel_z\":"); client.print(event.acceleration.z);
    client.println("}");

    delay(10);
    client.stop();
  }
}

