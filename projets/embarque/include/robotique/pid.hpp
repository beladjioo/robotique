// Régulateur PID embarqué — jumeau de projets/commande/src/commande/pid.py.
// Dérivée filtrée sur la mesure, saturation et anti-emballement par intégration conditionnelle.
#pragma once

namespace robotique {

struct PidGains {
  float kp = 0.0f;
  float ki = 0.0f;
  float kd = 0.0f;
};

class Pid {
 public:
  Pid(PidGains gains, float out_min, float out_max, float derivative_tau = 0.0f)
      : gains_(gains), out_min_(out_min), out_max_(out_max), derivative_tau_(derivative_tau) {}

  void reset() {
    integral_ = 0.0f;
    derivative_ = 0.0f;
    has_previous_ = false;
  }

  void set_gains(PidGains gains) { gains_ = gains; }
  float integral() const { return integral_; }

  float update(float setpoint, float measurement, float dt) {
    if (dt <= 0.0f) return clamp(gains_.kp * (setpoint - measurement) + integral_);
    const float error = setpoint - measurement;

    float raw_derivative = 0.0f;
    if (has_previous_) raw_derivative = -(measurement - previous_measurement_) / dt;
    previous_measurement_ = measurement;
    has_previous_ = true;
    const float alpha = dt / (derivative_tau_ + dt);
    derivative_ += alpha * (raw_derivative - derivative_);

    const float proportional = gains_.kp * error;
    const float derivative = gains_.kd * derivative_;
    const float candidate = integral_ + gains_.ki * error * dt;
    const float unsaturated = proportional + candidate + derivative;
    float output = clamp(unsaturated);
    const bool saturated = output != unsaturated;
    const bool helps = (unsaturated > out_max_ && error < 0.0f) || (unsaturated < out_min_ && error > 0.0f);
    if (!saturated || helps) {
      integral_ = candidate;
    } else {
      output = clamp(proportional + integral_ + derivative);
    }
    return output;
  }

 private:
  float clamp(float value) const {
    return value < out_min_ ? out_min_ : (value > out_max_ ? out_max_ : value);
  }

  PidGains gains_;
  float out_min_;
  float out_max_;
  float derivative_tau_;
  float integral_ = 0.0f;
  float derivative_ = 0.0f;
  float previous_measurement_ = 0.0f;
  bool has_previous_ = false;
};

}  // namespace robotique
