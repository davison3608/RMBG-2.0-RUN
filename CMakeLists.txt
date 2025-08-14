cmake_minimum_required(VERSION 3.10)
project(rmbggui)
enable_language(CXX CUDA)

# 强制启用位置无关代码（PIC）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_86,code=sm_86")

if(NOT DEFINED CUDA_ROOT)
    set(CUDA_ROOT $ENV{CUDA_ROOT} $ENV{CUDA_HOME} /usr/local/cuda)
endif()

# 查找依赖库
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package(Threads REQUIRED)

find_package(CUDA 11.0 REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib")
    link_directories(${CUDA_LIB_DIR})  
else()
    message(FATAL_ERROR "CUDA not found. Check CUDA_ROOT or CUDA_HOME environment variable.")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)

# 查找TensorRT
if(NOT DEFINED TENSORRT_ROOT)
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include 
          /usr/include 
          /usr/local/include 
          /usr/local/cuda/include 
          /opt/tensorrt/include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib 
          /usr/lib 
          /usr/local/lib 
          /usr/local/cuda/lib64 
          /opt/tensorrt/lib)

find_library(TENSORRT_PARSER nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib 
          /usr/lib 
          /usr/local/lib 
          /usr/local/cuda/lib64 
          /opt/tensorrt/lib)

find_library(TENSORRT_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT}/lib 
          /usr/lib 
          /usr/local/lib 
          /usr/local/cuda/lib64 
          /opt/tensorrt/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PARSER OR NOT TENSORRT_PLUGIN)
    message(FATAL_ERROR "TensorRT not found. Check TENSORRT_ROOT: ${TENSORRT_ROOT}")
endif()

# 源文件列表
set(LIB_SOURCES
    cpp/process.cpp
    cpp/qtasyncsendreceive.cpp
    cpp/qtcmd.cpp
    cpp/qtconnected.cpp
    cpp/qtgui.cpp
    cpp/serverasyncsendreceive.cpp
    cpp/servercmd.cpp
    cpp/serverconnected.cpp
    # CUDA源文件
    cpp/cvprocess.cu
    cpp/rtready.cu
    cpp/rtinfer.cu
)

# 头文件与UI文件
set(HEADERS
    include/cvpacked.h
    include/qtpacked.h
    include/rtpacketed.h
    include/serverpacked.h
    include/ui_cmddialog.h
    include/ui_detailsscreen.h
    include/ui_exitdialog.h
    include/ui_splashscreen.h
    include/ui_talkformscreen.h
)

set(FORMS
    file/cmddialog.ui
    file/detailsscreen.ui
    file/exitdialog.ui
    file/splashscreen.ui
    file/talkformscreen.ui
)

# 处理Qt UI和MOC
qt5_wrap_ui(UI_HEADERS ${FORMS})
qt5_wrap_cpp(MOC_SOURCES ${HEADERS})

# 生成共享库
cuda_add_library(rmbgcore SHARED
    ${LIB_SOURCES}
    ${HEADERS}
    ${UI_HEADERS}
    ${MOC_SOURCES}
)

set_target_properties(rmbgcore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)

target_link_libraries(rmbgcore
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Threads::Threads
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PARSER}
    ${TENSORRT_PLUGIN}
    cudart  
)

# 包含目录
target_include_directories(rmbgcore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}
    cpp/main.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(${PROJECT_NAME}
    rmbgcore
    ${CUDA_LIBRARIES}
    cudart  
)

# 安装规则
if(UNIX AND NOT ANDROID)
    install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION /opt/${PROJECT_NAME}/bin
    )
    install(TARGETS rmbgcore
        LIBRARY DESTINATION /opt/${PROJECT_NAME}/lib
    )
elseif(QNX)
    install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION /tmp/${PROJECT_NAME}/bin
    )
    install(TARGETS rmbgcore
        LIBRARY DESTINATION /tmp/${PROJECT_NAME}/lib
    )
endif()
    
