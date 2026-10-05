#include <cstdint>

#include "mini_test.hpp"
#include "robotique/diff_drive_odometry.hpp"
#include "robotique/hal.hpp"
#include "robotique/motor_speed_controller.hpp"
#include "robotique/pid.hpp"
#include "robotique/quadrature_decoder.hpp"
#include "robotique/spsc_queue.hpp"

using namespace robotique;

TEST(pid_proportionnel_et_saturation) {
  Pid pid({2.0f, 0.0f, 0.0f}, -1.0f, 1.0f);
  CHECK_NEAR(pid.update(0.25f, 0.0f, 0.01f), 0.5f, 1e-6);
  CHECK_NEAR(pid.update(10.0f, 0.0f, 0.01f), 1.0f, 1e-6);
}

TEST(pid_anti_emballement) {
  Pid pid({1.0f, 10.0f, 0.0f}, -1.0f, 1.0f);
  for (int i = 0; i < 1000; ++i) pid.update(100.0f, 0.0f, 0.01f);
  CHECK(pid.integral() <= 1.0f + 1e-5f);
  CHECK(pid.update(0.0f, 5.0f, 0.01f) < 0.0f);  // désaturation immédiate
}

TEST(pid_sans_pic_de_derivee) {
  Pid pid({0.0f, 0.0f, 1.0f}, -100.0f, 100.0f);
  pid.update(0.0f, 0.0f, 0.01f);
  CHECK_NEAR(pid.update(10.0f, 0.0f, 0.01f), 0.0f, 1e-6);
}

TEST(codeur_quadrature_sens_et_erreurs) {
  QuadratureDecoder decoder;
  // A en avance sur B : 00 -> 10 -> 11 -> 01 -> 00, soit 4 fronts positifs par période.
  const bool a_seq[] = {true, true, false, false};
  const bool b_seq[] = {false, true, true, false};
  for (int period = 0; period < 3; ++period)
    for (int k = 0; k < 4; ++k) decoder.update(a_seq[k], b_seq[k]);
  CHECK(decoder.count() == 12);
  for (int k = 3; k >= 0; --k) decoder.update(a_seq[(k + 3) % 4], b_seq[(k + 3) % 4]);
  CHECK(decoder.count() == 8);
  CHECK(decoder.errors() == 0);
  decoder.reset();
  decoder.update(false, false);
  decoder.update(true, true);  // transition impossible : les deux voies ont changé
  CHECK(decoder.errors() == 1);
  CHECK(decoder.count() == 0);
}

TEST(odometrie_ligne_droite_et_rotation) {
  const float radius = 0.033f, base = 0.16f, ticks = 1000.0f;
  DiffDriveOdometry odom(radius, base, ticks);
  odom.update(1000, 1000);  // un tour de roue de chaque côté
  CHECK_NEAR(odom.pose().x, 2.0 * 3.14159265 * radius, 1e-5);
  CHECK_NEAR(odom.pose().theta, 0.0, 1e-6);
  odom.reset();
  // Rotation sur place d'un demi-tour : chaque roue parcourt pi * base / 2.
  const int32_t half_turn = static_cast<int32_t>(0.5f * base / (2.0f * radius) * ticks + 0.5f);
  odom.update(-half_turn, half_turn);
  CHECK_NEAR(std::fabs(odom.pose().theta), 3.14159265, 2e-3);
  CHECK_NEAR(odom.pose().x, 0.0, 1e-6);
}

TEST(file_spsc) {
  SpscQueue<int, 4> queue;
  CHECK(queue.empty());
  CHECK(queue.push(1) && queue.push(2) && queue.push(3));
  CHECK(!queue.push(4));  // capacité utile : 3
  int value = 0;
  CHECK(queue.pop(value) && value == 1);
  CHECK(queue.push(4));
  CHECK(queue.pop(value) && value == 2);
  CHECK(queue.pop(value) && value == 3);
  CHECK(queue.pop(value) && value == 4);
  CHECK(!queue.pop(value));
}

// Moteur simulé (premier ordre) qui implémente aussi le codeur : test de bout en bout via la HAL.
class SimulatedMotor : public IMotor, public IEncoder {
 public:
  void set_command(float command) override { command_ = command; }
  int32_t count() const override { return static_cast<int32_t>(position_ * kCountsPerRad); }
  void step(float dt) {
    velocity_ += dt * (kGain * command_ - velocity_) / kTau;
    position_ += velocity_ * dt;
  }
  static constexpr float kCountsPerRev = 2048.0f;
  static constexpr float kCountsPerRad = kCountsPerRev / (2.0f * 3.14159265f);

 private:
  static constexpr float kGain = 40.0f;  // rad/s à pleine commande
  static constexpr float kTau = 0.05f;   // constante de temps mécanique (s)
  float command_ = 0.0f;
  double position_ = 0.0;
  float velocity_ = 0.0f;
};

TEST(asservissement_vitesse_bout_en_bout) {
  SimulatedMotor motor;
  MotorSpeedController controller(motor, motor, {0.05f, 1.0f, 0.0f}, SimulatedMotor::kCountsPerRev, 0.005f);
  controller.set_target(20.0f);
  const float dt = 0.001f;
  float command = 0.0f;
  for (int i = 0; i < 2000; ++i) {
    command = controller.update(dt);
    motor.step(dt);
  }
  CHECK_NEAR(controller.velocity(), 20.0f, 0.5f);
  CHECK(command > 0.0f && command < 1.0f);
  controller.stop();  // consigne nulle : le régulateur freine activement jusqu'à l'arrêt
  CHECK(controller.update(dt) < 0.0f);
  for (int i = 0; i < 2000; ++i) {
    controller.update(dt);
    motor.step(dt);
  }
  CHECK_NEAR(controller.velocity(), 0.0f, 0.5f);
}

int main() { return mini_test::run_all(); }
