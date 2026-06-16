cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
project(cpp-neural-network)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(BLAS REQUIRED)
find_package(Boost REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++23 -Wall -Wextra -Wformat -Wformat-security -Wc++20-compat -Wconversion -Wsign-conversion -Wsign-compare -Wuninitialized -Wno-uninitialized -Wreturn-local-addr -Wshadow -Wimplicit-fallthrough -Wold-style-cast")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native")

add_compile_options(-stdlib=libc++)
add_link_options(-stdlib=libc++)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE BLAS::BLAS)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::boost)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/debug")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/release")
endif()
