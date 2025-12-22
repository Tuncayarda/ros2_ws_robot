cmake_minimum_required(VERSION 3.8)
project(motor_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(std_msgs REQUIRED)
pkg_check_modules(GPIOD REQUIRED libgpiodcxx)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(motor_driver_node src/motor_driver_node.cpp)
target_include_directories(motor_driver_node PRIVATE ${GPIOD_INCLUDE_DIRS})
target_link_libraries(motor_driver_node ${GPIOD_LIBRARIES})
ament_target_dependencies(motor_driver_node rclcpp geometry_msgs std_msgs)

install(TARGETS motor_driver_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
