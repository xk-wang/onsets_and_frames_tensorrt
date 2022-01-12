cmake_minimum_required(VERSION 3.22)
project(midi)
add_definitions(-std=c++11)

file(GLOB MIDI_FILES ../../cpp/midifile/src/*.cpp)
add_executable(midi ${MIDI_FILES} ${PROJECT_SOURCE_DIR}/encode_midi.cpp)
target_include_directories(midi PRIVATE ../../cpp/midifile/include)