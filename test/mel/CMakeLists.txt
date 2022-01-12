# cmake needs this line
cmake_minimum_required(VERSION 3.1)
project(mel)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -mfma")

add_executable(mel test_mel.cpp)
target_include_directories(mel PRIVATE ../../cpp/eigen)
target_include_directories(mel PRIVATE ../../cpp/librosa/include)
target_link_libraries(mel PRIVATE sndfile)
target_link_libraries(mel PRIVATE samplerate)

add_executable(split split_audio.cpp)
target_link_libraries(split PRIVATE sndfile)