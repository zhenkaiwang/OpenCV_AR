cmake_minimum_required(VERSION 2.8) 
PROJECT (AugmentedReality)
find_package(OpenCV REQUIRED )
set( NAME_SRC
    src/AR.cpp    
)
set( NAME_HEADERS       

)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( AugmentedReality ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( AugmentedReality ${OpenCV_LIBS} )