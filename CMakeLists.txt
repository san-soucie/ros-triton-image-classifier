cmake_minimum_required(VERSION 2.8.3)
project(triton_image_classifier)

find_package(
  catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  Classification.msg
  ObjectHypothesisWithClassName.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS nodes/classifier
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
