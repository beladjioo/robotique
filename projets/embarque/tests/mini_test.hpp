// Micro-framework de test sans dépendance (portable vers des cibles sans gtest).
#pragma once

#include <cmath>
#include <cstdio>

namespace mini_test {

struct Registry {
  using Fn = void (*)();
  static constexpr int kMax = 64;
  const char* names[kMax];
  Fn tests[kMax];
  int count = 0;
  int failures = 0;

  static Registry& get() {
    static Registry registry;
    return registry;
  }
};

struct Registrar {
  Registrar(const char* name, Registry::Fn fn) {
    Registry& r = Registry::get();
    r.names[r.count] = name;
    r.tests[r.count] = fn;
    ++r.count;
  }
};

inline int run_all() {
  Registry& r = Registry::get();
  for (int i = 0; i < r.count; ++i) {
    const int before = r.failures;
    r.tests[i]();
    std::printf("[%s] %s\n", r.failures == before ? " OK " : "ECHEC", r.names[i]);
  }
  std::printf("%d test(s), %d échec(s)\n", r.count, r.failures);
  return r.failures == 0 ? 0 : 1;
}

}  // namespace mini_test

#define TEST(name)                                                   \
  static void name();                                                \
  static const mini_test::Registrar name##_registrar(#name, &name);  \
  static void name()

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::printf("  %s:%d : échec de CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      ++mini_test::Registry::get().failures;                                     \
    }                                                                            \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= (tol))
