cmake_minimum_required(VERSION 2.6)
project( spine_recognize )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories (include)
add_executable( spine_recognize 
    src/spine_recognize.cpp 
    include/spine_recognize.h)
target_link_libraries( spine_recognize ${OpenCV_LIBS} lept tesseract)
