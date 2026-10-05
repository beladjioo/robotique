// Asservissement en vitesse d'un moteur : codeur -> estimation de vitesse -> PID -> moteur.
#pragma once

#include <cstdint>

#include "robotique/hal.hpp"
#include "robotique/pid.hpp"

namespace robotique {

class MotorSpeedController {
 public:
  MotorSpeedController(IMotor& motor, const IEncoder& encoder, PidGains gains, float counts_per_revolution,
                       float velocity_filter_tau = 0.0f)
      : motor_(motor),
        encoder_(encoder),
        pid_(gains, -1.0f, 1.0f),
        radians_per_count_(2.0f * 3.14159265358979f / counts_per_revolution),
        velocity_filter_tau_(velocity_filter_tau),
        last_count_(encoder.count()) {}

  void set_target(float radians_per_second) { target_ = radians_per_second; }
  float velocity() const { return velocity_; }

  // À appeler à fréquence fixe (ex. 1 kHz depuis un timer). Retourne la commande appliquée.
  float update(float dt) {
    const int32_t count = encoder_.count();
    const float raw = static_cast<float>(count - last_count_) * radians_per_count_ / dt;
    last_count_ = count;
    const float alpha = dt / (velocity_filter_tau_ + dt);
    velocity_ += alpha * (raw - velocity_);
    const float command = pid_.update(target_, velocity_, dt);
    motor_.set_command(command);
    return command;
  }

  void stop() {
    target_ = 0.0f;
    pid_.reset();
    motor_.set_command(0.0f);
  }

 private:
  IMotor& motor_;
  const IEncoder& encoder_;
  Pid pid_;
  float radians_per_count_;
  float velocity_filter_tau_;
  int32_t last_count_;
  float target_ = 0.0f;
  float velocity_ = 0.0f;
};

}  // namespace robotique
