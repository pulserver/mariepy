cmake_minimum_required(VERSION 3.18)
project(mariepy_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)
find_package(Threads REQUIRED)

# One extension module holding every compiled kernel. List the sources
# explicitly: a glob makes a new file compile without anyone deciding it should.
pybind11_add_module(_ext
    ${CMAKE_SOURCE_DIR}/src/cpp/module.cpp
)

target_include_directories(_ext PRIVATE ${CMAKE_SOURCE_DIR}/src/cpp)

# The kernels partition independent work across standard C++ threads.
target_link_libraries(_ext PRIVATE Threads::Threads)

# Built into the build tree, never beside the sources: the wheel takes the
# module from the install step, so a binary left in the source tree would be
# swept into wheels built for every other interpreter too.
set_target_properties(_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/mariepy"
)

install(TARGETS _ext DESTINATION mariepy)
