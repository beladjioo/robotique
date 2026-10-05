// File circulaire sans verrou, un producteur / un consommateur (ex. : interruption -> boucle principale).
#pragma once

#include <atomic>
#include <cstddef>

namespace robotique {

template <typename T, std::size_t Capacity>
class SpscQueue {
  static_assert(Capacity >= 2, "capacité minimale : 2");

 public:
  bool push(const T& value) {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    const std::size_t next = (head + 1) % Capacity;
    if (next == tail_.load(std::memory_order_acquire)) return false;  // pleine
    buffer_[head] = value;
    head_.store(next, std::memory_order_release);
    return true;
  }

  bool pop(T& value) {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) return false;  // vide
    value = buffer_[tail];
    tail_.store((tail + 1) % Capacity, std::memory_order_release);
    return true;
  }

  bool empty() const { return head_.load() == tail_.load(); }
  // Une case reste toujours libre pour distinguer « pleine » de « vide ».
  static constexpr std::size_t capacity() { return Capacity - 1; }

 private:
  T buffer_[Capacity]{};
  std::atomic<std::size_t> head_{0};
  std::atomic<std::size_t> tail_{0};
};

}  // namespace robotique
