// Odométrie d'un robot à entraînement différentiel à partir des tops codeurs.
#pragma once

#include <cmath>
#include <cstdint>

namespace robotique {

struct Pose2D {
  float x = 0.0f;
  float y = 0.0f;
  float theta = 0.0f;
};

class DiffDriveOdometry {
 public:
  DiffDriveOdometry(float wheel_radius, float wheel_base, float ticks_per_revolution)
      : meters_per_tick_(2.0f * kPi * wheel_radius / ticks_per_revolution), wheel_base_(wheel_base) {}

  // Intègre les incréments de tops depuis le dernier appel (intégration au point milieu).
  void update(int32_t delta_left_ticks, int32_t delta_right_ticks) {
    const float left = static_cast<float>(delta_left_ticks) * meters_per_tick_;
    const float right = static_cast<float>(delta_right_ticks) * meters_per_tick_;
    const float distance = 0.5f * (left + right);
    const float rotation = (right - left) / wheel_base_;
    const float heading = pose_.theta + 0.5f * rotation;
    pose_.x += distance * std::cos(heading);
    pose_.y += distance * std::sin(heading);
    pose_.theta = wrap(pose_.theta + rotation);
  }

  const Pose2D& pose() const { return pose_; }
  void reset(Pose2D pose = {}) { pose_ = pose; }

 private:
  static constexpr float kPi = 3.14159265358979f;
  static float wrap(float angle) {
    while (angle >= kPi) angle -= 2.0f * kPi;
    while (angle < -kPi) angle += 2.0f * kPi;
    return angle;
  }

  float meters_per_tick_;
  float wheel_base_;
  Pose2D pose_{};
};

}  // namespace robotique
