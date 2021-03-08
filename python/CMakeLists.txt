
set(PYBIND11_PYTHON_VERSION 3.7) 
add_subdirectory(pybind11)


set(OpenCV_DIR /home/ztq/software/my_opencv/opencv-3.3.0/release/opencv3.3/share/OpenCV/)
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
set(OpenCV_SHARED OFF)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
#include_directories(${OpenCV_INCLUDE_DIRS}) # only very old opencv requires this
message(STATUS "OpenCV found (${OpenCV_LIBS})")


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(dep_libs 
    ${OpenCV_LIBS}
    ncnn_lib
    )
set(lib_name bfq)
pybind11_add_module(${lib_name} ./lession_11.cpp)
   #    target_compile_options(${lib_name} PRIVATE -fpic)
target_link_libraries(${lib_name} PRIVATE ${dep_libs})
set_target_properties(${lib_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${lib_name} PROPERTY FOLDER "python")
