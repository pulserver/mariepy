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
    ${CMAKE_SOURCE_DIR}/src/cpp/convolution.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/coupling.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/volume.cpp
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

# The DIRECTFN singular integrals, as distributed in MARIE 3.0, carry an LGPL
# notice. They build as their own module so that the notice stays with them and
# never reaches the MIT sources in `_ext`. See src/cpp_lgpl/NOTICE.md.
#
# The voxel family links as its own sources stand. The three RWG sources each
# define the same helpers at file scope, so each is compiled through a wrapper
# that gives it a namespace; the wrappers, not the sources, are listed here.
file(GLOB DIRECTFN_VOXEL_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/directfn_vie/src_cpp/*.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/directfn_vie/src_cpp/WS_EA/*.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/directfn_vie/src_cpp/WS_ST/*.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/directfn_vie/src_cpp/WS_VA/*.cpp
)

pybind11_add_module(_directfn
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/module_directfn.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/rwg_namespace_st.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/rwg_namespace_ea.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/rwg_namespace_va.cpp
    ${DIRECTFN_VOXEL_SOURCES}
)

target_include_directories(_directfn PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/directfn_vie/include
    ${CMAKE_SOURCE_DIR}/src/cpp_lgpl/directfn_vie/src_cpp
)

# The vendored sources are kept byte-identical, warnings and all.
if(NOT MSVC)
    target_compile_options(_directfn PRIVATE -w)
endif()

set_target_properties(_directfn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/mariepy"
)

install(TARGETS _directfn DESTINATION mariepy)
