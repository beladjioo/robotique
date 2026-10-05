// Asservissement en vitesse d'un moteur CC avec codeur — exemple Arduino (Uno/Nano/ESP32).
// Copier le dossier include/robotique dans le dossier du croquis (ou dans libraries/) avant compilation.
// Câblage type : pont en H (PWM + DIR), codeur A/B sur broches d'interruption.

#include "robotique/motor_speed_controller.hpp"
#include "robotique/quadrature_decoder.hpp"

constexpr int kPinPwm = 9;
constexpr int kPinDir = 8;
constexpr int kPinEncoderA = 2;
constexpr int kPinEncoderB = 3;
constexpr float kCountsPerRev = 4.0f * 12.0f * 30.0f;  // 12 CPR x réducteur 30:1, décodage x4
constexpr unsigned long kPeriodUs = 1000;              // boucle de commande à 1 kHz

robotique::QuadratureDecoder decoder;

void onEncoderEdge() { decoder.update(digitalRead(kPinEncoderA), digitalRead(kPinEncoderB)); }

class ArduinoMotor : public robotique::IMotor {
 public:
  void set_command(float command) override {
    digitalWrite(kPinDir, command >= 0.0f ? HIGH : LOW);
    const float magnitude = command >= 0.0f ? command : -command;
    analogWrite(kPinPwm, static_cast<int>(magnitude * 255.0f));
  }
};

class ArduinoEncoder : public robotique::IEncoder {
 public:
  int32_t count() const override {
    noInterrupts();  // lecture atomique du compteur 32 bits sur AVR
    const int32_t value = decoder.count();
    interrupts();
    return value;
  }
};

ArduinoMotor motor;
ArduinoEncoder encoder;
robotique::MotorSpeedController controller(motor, encoder, {0.05f, 1.0f, 0.0f}, kCountsPerRev, 0.01f);
unsigned long last_update = 0;

void setup() {
  pinMode(kPinPwm, OUTPUT);
  pinMode(kPinDir, OUTPUT);
  pinMode(kPinEncoderA, INPUT_PULLUP);
  pinMode(kPinEncoderB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(kPinEncoderA), onEncoderEdge, CHANGE);
  attachInterrupt(digitalPinToInterrupt(kPinEncoderB), onEncoderEdge, CHANGE);
  Serial.begin(115200);
  controller.set_target(10.0f);  // rad/s
}

void loop() {
  const unsigned long now = micros();
  if (now - last_update >= kPeriodUs) {
    const float dt = (now - last_update) * 1e-6f;
    last_update = now;
    controller.update(dt);
  }
  static unsigned long last_print = 0;
  if (millis() - last_print > 100) {
    last_print = millis();
    Serial.println(controller.velocity());
  }
}
