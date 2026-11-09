cmake_minimum_required(VERSION 3.16)
project(opendoge_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(OPENDOGE_ENABLE_ONNX "Build ONNX Runtime policy backend when headers/libs are available" ON)

add_library(opendoge_deploy_core
  src/cli.cpp
  src/controller.cpp
  ../hardware/motor/el05_socketcan.cpp
  src/observer.cpp
  src/policy.cpp
  src/runtime_io.cpp
  src/safety.cpp
  src/status.cpp
)

target_include_directories(opendoge_deploy_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../hardware/motor>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(opendoge_deploy_core PRIVATE -Wall -Wextra -Wpedantic)

if(OPENDOGE_ENABLE_ONNX)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      $ENV{ONNXRUNTIME_ROOT}/include
      /usr/local/include
      /usr/include
  )
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS
      $ENV{ONNXRUNTIME_ROOT}/lib
      /usr/local/lib
      /usr/lib
      /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
  )
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    target_sources(opendoge_deploy_core PRIVATE src/onnx_policy.cpp)
    target_include_directories(opendoge_deploy_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(opendoge_deploy_core PUBLIC ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(opendoge_deploy_core PUBLIC OPENDOGE_HAS_ONNX=1)
  else()
    message(WARNING "ONNX Runtime not found; opendoge_deploy will build without ONNX backend")
  endif()
endif()

add_executable(opendoge_deploy src/main.cpp)
target_link_libraries(opendoge_deploy PRIVATE opendoge_deploy_core pthread)
target_compile_options(opendoge_deploy PRIVATE -Wall -Wextra -Wpedantic)

if(ONNXRUNTIME_LIBRARY_DIR)
  set_target_properties(opendoge_deploy PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
  )
endif()

install(TARGETS opendoge_deploy opendoge_deploy_core
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY configs/ DESTINATION share/${PROJECT_NAME}/configs)
