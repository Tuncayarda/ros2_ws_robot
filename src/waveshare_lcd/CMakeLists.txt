cmake_minimum_required(VERSION 3.8)
project(waveshare_lcd)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

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

add_executable(lcd_node src/lcd_node.cpp)
ament_target_dependencies(lcd_node
  rclcpp
  std_msgs
  sensor_msgs
)

target_link_libraries(lcd_node gpiod)
target_link_libraries(lcd_node gpiod ${OpenCV_LIBS})
target_include_directories(lcd_node PRIVATE ${OpenCV_INCLUDE_DIRS})

install(TARGETS lcd_node
  DESTINATION lib/${PROJECT_NAME})


ament_package()
