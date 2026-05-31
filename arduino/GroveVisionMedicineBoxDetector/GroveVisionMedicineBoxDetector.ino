#include <Seeed_Arduino_SSCMA.h>

// Grove Vision AI V2 uses SSCMA AT over UART at 921600 baud by default.
// Use a board with a hardware UART for the Grove Vision module.

#ifdef ESP32
#include <HardwareSerial.h>
HardwareSerial atSerial(0);
#else
#define atSerial Serial1
#endif

// Set this to the class id of your medicine-box model.
// Leave it as -1 to print every detected box.
const int MEDICINE_BOX_TARGET = -1;
const int MIN_SCORE = 50;

SSCMA AI;

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
        delay(10);
    }

    Serial.println("Starting Grove Vision AI...");
    if (!AI.begin(&atSerial)) {
        Serial.println("Grove Vision AI begin failed. Check UART wiring and power.");
        while (1) {
            delay(1000);
        }
    }
    Serial.println("Grove Vision AI ready.");
}

void loop() {
    int ret = AI.invoke(1, false, false);
    if (ret != 0) {
        Serial.print("invoke failed: ");
        Serial.println(ret);
        delay(500);
        return;
    }

    bool found = false;
    for (int i = 0; i < AI.boxes().size(); i++) {
        const auto &box = AI.boxes()[i];
        if (box.score < MIN_SCORE) {
            continue;
        }
        if (MEDICINE_BOX_TARGET >= 0 && box.target != MEDICINE_BOX_TARGET) {
            continue;
        }

        found = true;
        int left = box.x - box.w / 2;
        int top = box.y - box.h / 2;
        int right = box.x + box.w / 2;
        int bottom = box.y + box.h / 2;

        Serial.print("medicine_box target=");
        Serial.print(box.target);
        Serial.print(" score=");
        Serial.print(box.score);
        Serial.print(" center=(");
        Serial.print(box.x);
        Serial.print(",");
        Serial.print(box.y);
        Serial.print(") size=(");
        Serial.print(box.w);
        Serial.print("x");
        Serial.print(box.h);
        Serial.print(") rect=(");
        Serial.print(left);
        Serial.print(",");
        Serial.print(top);
        Serial.print(")-(");
        Serial.print(right);
        Serial.print(",");
        Serial.print(bottom);
        Serial.println(")");
    }

    if (!found) {
        Serial.println("no medicine_box box");
    }

    delay(200);
}
