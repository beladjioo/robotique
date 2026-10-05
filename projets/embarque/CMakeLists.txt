cmake_minimum_required(VERSION 3.16)
project(robotique_embarque LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Bibliothèque « header-only » : aucune allocation dynamique, aucune exception,
# utilisable telle quelle sur microcontrôleur (Arduino, ESP32, STM32...).
add_library(robotique_embarque INTERFACE)
target_include_directories(robotique_embarque INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(ROBOTIQUE_EMBARQUE_TESTS "Compiler les tests unitaires sur l'hôte" ON)
if(ROBOTIQUE_EMBARQUE_TESTS)
  enable_testing()
  add_executable(test_embarque tests/test_main.cpp)
  target_link_libraries(test_embarque PRIVATE robotique_embarque)
  if(MSVC)
    target_compile_options(test_embarque PRIVATE /W4 /WX)
  else()
    target_compile_options(test_embarque PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  add_test(NAME test_embarque COMMAND test_embarque)
endif()
