cmake_minimum_required(VERSION 3.16)
project(bunker_offline_localization VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common io)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

set(SMALL_GICP_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/small_gicp")
if(NOT EXISTS "${SMALL_GICP_VENDOR_DIR}/include/small_gicp/registration/registration.hpp")
  message(FATAL_ERROR
    "Official small_gicp submodule is missing. Run: git submodule update --init --recursive")
endif()

add_library(small_gicp_vendor INTERFACE)
target_include_directories(small_gicp_vendor SYSTEM INTERFACE
  "${SMALL_GICP_VENDOR_DIR}/include")
target_link_libraries(small_gicp_vendor INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX)

add_library(${PROJECT_NAME}_core
  src/map_loader.cpp
  src/metrics.cpp
  src/reference_trajectory.cpp
  src/registration.cpp
  src/scan_preprocessor.cpp
  src/transforms.cpp)
target_include_directories(${PROJECT_NAME}_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME}_core SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME}_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(${PROJECT_NAME}_core
  Eigen3::Eigen small_gicp_vendor ${PCL_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}_core sensor_msgs)

add_executable(covariance_adapter_node src/covariance_adapter_node.cpp)
target_include_directories(covariance_adapter_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(covariance_adapter_node PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(covariance_adapter_node nav_msgs rclcpp sensor_msgs)

add_executable(offline_localizer_node src/offline_localizer_node.cpp)
target_compile_options(offline_localizer_node PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(offline_localizer_node ${PROJECT_NAME}_core)
ament_target_dependencies(offline_localizer_node nav_msgs rclcpp sensor_msgs)

install(TARGETS
  ${PROJECT_NAME}_core
  covariance_adapter_node
  offline_localizer_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/generate_independent_report.py
  scripts/generate_report.py
  scripts/run_independent_163346.sh
  scripts/run_offline_smoke_test.sh
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(transform_convention_test test/transform_convention_test.cpp)
  target_link_libraries(transform_convention_test ${PROJECT_NAME}_core)

  ament_add_gtest(sign_convention_test test/sign_convention_test.cpp)
  target_link_libraries(sign_convention_test ${PROJECT_NAME}_core)

  ament_add_gtest(reference_trajectory_test test/reference_trajectory_test.cpp)
  target_link_libraries(reference_trajectory_test ${PROJECT_NAME}_core)

  ament_add_gtest(metrics_test test/metrics_test.cpp)
  target_link_libraries(metrics_test ${PROJECT_NAME}_core)

  ament_add_gtest(registration_smoke_test test/registration_smoke_test.cpp)
  target_link_libraries(registration_smoke_test ${PROJECT_NAME}_core)

  ament_add_gtest(time_window_test test/time_window_test.cpp)
  target_link_libraries(time_window_test ${PROJECT_NAME}_core)
endif()

ament_export_include_directories(include)
ament_export_dependencies(Eigen3 sensor_msgs)
ament_package()
