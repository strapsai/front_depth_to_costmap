cmake_minimum_required(VERSION 3.8)
project(depth_to_pointcloud_pub)

set(Python3_EXECUTABLE /usr/bin/python3.10)

find_package(pybind11 REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(nav2_costmap_2d REQUIRED)

pybind11_add_module(costmap_2d_py src/costmap_python.cpp)

# C++ 코드가 nav2_costmap_2d의 헤더 파일을 찾을 수 있도록 경로를 설정합니다.
target_include_directories(costmap_2d_py PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${nav2_costmap_2d_INCLUDE_DIRS}
)

# 컴파일된 C++ 모듈(.so 파일)이 어떤 라이브러리를 필요로 하는지 명시합니다.
# nav2_costmap_2d가 사용하는 라이브러리들을 연결해줍니다.
target_link_libraries(costmap_2d_py PUBLIC ${nav2_costmap_2d_LIBRARIES})

# 빌드된 파이썬 모듈을 올바른 위치에 설치하여
# 파이썬 코드에서 'import costmap_2d_py'가 가능하도록 설정합니다.
install(TARGETS costmap_2d_py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})


install(PROGRAMS
        depth_to_pointcloud_pub/depth_to_pointcloud_node.py
        DESTINATION lib/${PROJECT_NAME})

# ament_python 패키지에서 정의된 파이썬 관련 설정을 실행합니다.
# (setup.py를 처리하여 파이썬 노드를 설치하는 부분)
ament_python_install_package(${PROJECT_NAME})

# ament 패키징을 마무리합니다.
ament_package()

