cmake_minimum_required(VERSION 3.15...3.31)
project(onlinete_native LANGUAGES CXX)

if (NOT SKBUILD)
  message(FATAL_ERROR "Build this extension through pip/scikit-build-core (for example: pip install -e .)")
endif()

find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

nanobind_add_module(_path_kernels STABLE_ABI native/path_kernels.cpp)
target_compile_features(_path_kernels PRIVATE cxx_std_17)
if (MINGW)
  # MinGW's OpenMP runtime is dynamic. Bundle its small runtime closure next
  # to the extension so editable installs and wheels are self-contained.
  get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  target_link_libraries(_path_kernels PRIVATE OpenMP::OpenMP_CXX)
  target_link_options(_path_kernels PRIVATE -static-libgcc -static-libstdc++)
  set(MINGW_OPENMP_DLLS
    "${MINGW_BIN_DIR}/libgomp-1.dll"
    "${MINGW_BIN_DIR}/libdl.dll"
    "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
    "${MINGW_BIN_DIR}/libwinpthread-1.dll"
  )
else()
  target_link_libraries(_path_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()

install(
  TARGETS _path_kernels
  LIBRARY DESTINATION te/algorithms/formulations/path_based/distributed
  RUNTIME DESTINATION te/algorithms/formulations/path_based/distributed
)
if (MINGW)
  install(
    FILES ${MINGW_OPENMP_DLLS}
    DESTINATION te/algorithms/formulations/path_based/distributed
  )
endif()
