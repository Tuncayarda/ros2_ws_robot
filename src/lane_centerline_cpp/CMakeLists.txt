cmake_minimum_required(VERSION 3.8)
project(lane_centerline_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(lane_centerline_node src/lane_centerline_node.cpp)
ament_target_dependencies(lane_centerline_node rclcpp sensor_msgs std_msgs)
target_link_libraries(lane_centerline_node ${OpenCV_LIBS})

install(TARGETS lane_centerline_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
