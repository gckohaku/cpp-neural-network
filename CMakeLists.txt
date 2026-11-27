cmake_minimum_required(VERSION 3.10)
set(TEST_FILE_NAME "cpp-neural-network_test")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_CXX_FLAGS "-std=c++23 -Wall -Wextra -Wformat -Wformat-security -Wconversion -Wsign-conversion -Wsign-compare -Wuninitialized -Wreturn-local-addr -Wshadow -Wimplicit-fallthrough -Wold-style-cast")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native")
add_compile_options(-stdlib=libc++)
add_link_options(-stdlib=libc++)

project(cpp-neural-network)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenBLAS 0.3.34 CONFIG REQUIRED)
# for test
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(${PROJECT_NAME} src/main.cpp)
# for test
add_executable(${TEST_FILE_NAME} tests/test_main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# for test
target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS Boost::headers)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
# for test 2 lines
target_include_directories(${TEST_FILE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TEST_FILE_NAME} PRIVATE OpenBLAS::OpenBLAS Boost::headers)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/debug")
    # for test 2 lines 
    set_target_properties(${TEST_FILE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/debug")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/release")
endif()
