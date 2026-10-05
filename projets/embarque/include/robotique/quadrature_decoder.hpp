// Décodage d'un codeur incrémental en quadrature (voies A et B), appelable depuis une interruption.
#pragma once

#include <cstdint>

namespace robotique {

class QuadratureDecoder {
 public:
  // À appeler à chaque changement d'état des voies (ou périodiquement, assez vite).
  // Sens positif : la voie A est en avance sur la voie B.
  void update(bool a, bool b) {
    const uint8_t state = static_cast<uint8_t>((a ? 2u : 0u) | (b ? 1u : 0u));
    const uint8_t index = static_cast<uint8_t>((previous_ << 2) | state);
    const int8_t delta = kTable[index];
    // Les deux voies ont changé en même temps : impulsion manquée (vitesse trop élevée ou bruit).
    if (delta == 0 && state != previous_) ++errors_;
    count_ += delta;
    previous_ = state;
  }

  int32_t count() const { return count_; }
  uint32_t errors() const { return errors_; }
  void reset(int32_t count = 0) { count_ = count; }

 private:
  static constexpr int8_t kTable[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
  uint8_t previous_ = 0;
  int32_t count_ = 0;
  uint32_t errors_ = 0;
};

}  // namespace robotique
