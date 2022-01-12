cmake_minimum_required(VERSION 3.22)
project(amt)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -mavx -mfma")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# include and link dirs, you need adapt them if yours are different
# cuda
include_directories(/home/wxk/environment/cuda10.2/include)
link_directories(/home/wxk/environment/cuda10.2/lib64)
# cudnn
include_directories(/home/wxk/environment/cudnn-10.2-linux-x64-v8.2.2.26/include)
link_directories(/home/wxk/environment/cudnn-10.2-linux-x64-v8.2.2.26/lib64)
# tensorrt
include_directories(/home/wxk/environment/TensorRT-8.0.1.6/include)
link_directories(/home/wxk/environment/TensorRT-8.0.1.6/lib)
# opencv
set(OpenCV_DIR /home/wxk/environment/opencv-4.2.0/lib64/cmake/opencv4)
set(OpenCV_LIBRARY_DIRS /home/wxk/environment/opencv-4.2.0/lib64)
find_package(OpenCV 4.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
# cnpy
include_directories(/home/wxk/environment/cnpy/include)
link_directories(/home/wxk/environment/cnpy/lib)

# eigen
include_directories(eigen)
# librosa
include_directories(librosa/include)
# midifile
include_directories(midifile/include)
file(GLOB MIDI_FILES midifile/src/*.cpp)

add_executable(amt ${MIDI_FILES} ${PROJECT_SOURCE_DIR}/amt.cpp)
target_link_libraries(amt ${OpenCV_LIBS})
target_link_libraries(amt nvinfer)
target_link_libraries(amt cudart)

target_link_libraries(amt sndfile)
target_link_libraries(amt samplerate)
target_link_libraries(amt cnpy)

# boost filesystem
target_link_libraries(amt boost_filesystem)
target_link_libraries(amt boost_system)