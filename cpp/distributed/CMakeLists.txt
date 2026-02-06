cmake_minimum_required(VERSION 3.5)
project(dist-mnist)

find_package(Torch REQUIRED)

set(MPI_C_COMPILER /usr/lib64/openmpi/bin/mpicc)
set(MPI_CXX_COMPILER /usr/lib64/openmpi/bin/mpicxx)
find_package(MPI REQUIRED)

message(STATUS "MPI_C_INCLUDE_PATH: ${MPI_C_INCLUDE_PATH}")
message(STATUS "MPI_CXX_INCLUDE_PATH: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(dist-mnist dist-mnist.cpp)
target_link_libraries(dist-mnist ${TORCH_LIBRARIES})
target_link_libraries(dist-mnist ${MPI_LIBRARIES})
target_compile_definitions(dist-mnist PRIVATE USE_C10D_MPI)
target_compile_definitions(dist-mnist PRIVATE USE_C10D_NCCL)

# Tensor Parallel Inference example
add_executable(tp-demo-infer tp_demo_infer.cpp)
target_link_libraries(tp-demo-infer ${TORCH_LIBRARIES})
target_link_libraries(tp-demo-infer ${MPI_LIBRARIES})
target_compile_definitions(tp-demo-infer PRIVATE USE_C10D_MPI)

# Sequence Parallel Inference example (Ulysses)
add_executable(sp-demo-infer sp_demo_infer.cpp)
target_link_libraries(sp-demo-infer ${TORCH_LIBRARIES})
target_link_libraries(sp-demo-infer ${MPI_LIBRARIES})
target_compile_definitions(sp-demo-infer PRIVATE USE_C10D_MPI)

add_executable(torch-infer torch_infer.cpp)
target_link_libraries(torch-infer ${TORCH_LIBRARIES})
target_link_libraries(torch-infer ${MPI_LIBRARIES})

# Ulysses Sequence Parallelism with TorchScript
add_executable(torch-infer-ulysses-sp torch_infer_ulysses_sp.cpp)
target_link_libraries(torch-infer-ulysses-sp ${TORCH_LIBRARIES})
target_link_libraries(torch-infer-ulysses-sp ${MPI_LIBRARIES})
target_compile_definitions(torch-infer-ulysses-sp PRIVATE USE_C10D_MPI)

if(MPI_COMPILE_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(tp-demo-infer PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(sp-demo-infer PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dist-mnist PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(tp-demo-infer PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(sp-demo-infer PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
