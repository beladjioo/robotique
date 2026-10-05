// Couche d'abstraction matérielle (HAL) : la logique de commande ne connaît que ces interfaces.
// Chaque cible (Arduino, ESP-IDF, STM32 HAL, simulateur sur PC) fournit ses implémentations.
#pragma once

#include <cstdint>

namespace robotique {

class IMotor {
 public:
  virtual ~IMotor() = default;
  // Commande normalisée dans [-1, 1] (rapport cyclique PWM signé).
  virtual void set_command(float command) = 0;
};

class IEncoder {
 public:
  virtual ~IEncoder() = default;
  virtual int32_t count() const = 0;
};

}  // namespace robotique
