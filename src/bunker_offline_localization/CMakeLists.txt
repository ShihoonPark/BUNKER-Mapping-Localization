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
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

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
  src/accepted_pose_predictor.cpp
  src/coarse_global_initializer.cpp
  src/diagnostic_utils.cpp
  src/gicp_diagnostics.cpp
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
ament_target_dependencies(${PROJECT_NAME}_core
  geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs tf2_ros visualization_msgs)

add_executable(covariance_adapter_node src/covariance_adapter_node.cpp)
target_include_directories(covariance_adapter_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(covariance_adapter_node PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(covariance_adapter_node nav_msgs rclcpp sensor_msgs)

add_executable(coarse_global_initializer_node src/coarse_global_initializer_node.cpp)
target_compile_options(coarse_global_initializer_node PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(coarse_global_initializer_node ${PROJECT_NAME}_core)
ament_target_dependencies(coarse_global_initializer_node rclcpp sensor_msgs)

add_executable(offline_localizer_node src/offline_localizer_node.cpp)
target_compile_options(offline_localizer_node PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(offline_localizer_node ${PROJECT_NAME}_core)
ament_target_dependencies(offline_localizer_node geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs)

add_executable(filter_timing_monitor_node src/filter_timing_monitor_node.cpp)
target_compile_options(filter_timing_monitor_node PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(filter_timing_monitor_node geometry_msgs nav_msgs rclcpp sensor_msgs)

install(TARGETS
  ${PROJECT_NAME}_core
  coarse_global_initializer_node
  covariance_adapter_node
  filter_timing_monitor_node
  offline_localizer_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/generate_filter_comparison.py
  scripts/generate_full_bag_report.py
  scripts/generate_gicp_vertical_observability_diagnostic.py
  scripts/generate_gicp_correspondence_rviz_diagnostic.py
  scripts/generate_imu_gicp_physical_gate.py
  scripts/generate_independent_report.py
  scripts/generate_plateau_label_diagnostic.py
  scripts/generate_planar_ekf_gate_report.py
  scripts/generate_ply_support_surface_diagnostic.py
  scripts/generate_report.py
  scripts/generate_surface_pose_z_diagnostic.py
  scripts/run_filter_comparison.sh
  scripts/run_gicp_vertical_observability_diagnostic.sh
  scripts/run_gicp_correspondence_runtime_gate.sh
  scripts/verify_gicp_rviz_topics.py
  scripts/run_imu_gicp_physical_gate.sh
  scripts/run_independent_163346.sh
  scripts/run_offline_smoke_test.sh
  scripts/run_plateau_label_diagnostic.sh
  scripts/run_planar_ekf_gate.sh
  scripts/run_ply_support_surface_diagnostic.sh
  scripts/run_surface_pose_z_diagnostic.sh
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_test REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

  ament_add_gtest(coarse_global_initializer_test test/coarse_global_initializer_test.cpp)
  target_link_libraries(coarse_global_initializer_test ${PROJECT_NAME}_core)

  ament_add_gtest(time_window_test test/time_window_test.cpp)
  target_link_libraries(time_window_test ${PROJECT_NAME}_core)

  ament_add_gtest(prediction_policy_test test/prediction_policy_test.cpp)
  target_link_libraries(prediction_policy_test ${PROJECT_NAME}_core)

  ament_add_gtest(gicp_diagnostics_test test/gicp_diagnostics_test.cpp)
  target_link_libraries(gicp_diagnostics_test ${PROJECT_NAME}_core)

  ament_add_test(
    filter_comparison_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/filter_comparison_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    planar_ekf_gate_report_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/planar_ekf_gate_report_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    imu_gicp_physical_gate_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/imu_gicp_physical_gate_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    plateau_label_diagnostic_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/plateau_label_diagnostic_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    surface_pose_z_diagnostic_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/surface_pose_z_diagnostic_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    ply_support_surface_diagnostic_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ply_support_surface_diagnostic_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    gicp_vertical_observability_diagnostic_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/gicp_vertical_observability_diagnostic_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    gicp_correspondence_rviz_diagnostic_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/gicp_correspondence_rviz_diagnostic_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    bag_d_relocalization_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/bag_d_relocalization_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")

  ament_add_test(
    full_bag_report_test
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}/test/full_bag_report_test.py
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")
endif()

ament_export_include_directories(include)
ament_export_dependencies(
  Eigen3 geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs tf2_ros visualization_msgs)
ament_package()
