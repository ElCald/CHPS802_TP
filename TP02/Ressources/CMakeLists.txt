# CMake minimal requiered version
cmake_minimum_required(VERSION 3.18)

# TP2
project(TP2 LANGUAGES CXX CUDA)

# Fichier à executer
set(EXECUTABLES prac2_1 prac2_2 prac2_device prac2_average prac2_average_test)


set(SRC_FOLDER "src") # Répertoire source
set(EXT_FILE ".cu") # Extention du fichier à compiler (.c, .cpp, .cu, etc)


# Turn on warnings for the compilator
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# # Set cuda architectures for my 3070 ti and my 4050
# set(CMAKE_CUDA_ARCHITECTURES 80)

# Include headers
include_directories(../../include)

# Add source files tous les fichiers nécessaires à la compilation
file(GLOB_RECURSE SOURCE_FILES ${SRC_FOLDER}/*.cpp ${SRC_FOLDER}/*.cu)

# Boucle pour créer chaque exécutable
foreach(EXE ${EXECUTABLES})
    add_executable(${EXE} ${SRC_FOLDER}/${EXE}${EXT_FILE} ${SRC_FOLDER}/${EXE}${EXT_FILE})
    set_target_properties(${EXE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${EXE} PRIVATE curand)
endforeach()


################
#Unitary tests
###############
enable_testing()


add_subdirectory(/home/elcalderonymora/googletest ${CMAKE_BINARY_DIR}/googletest_build)


# Link with GoogleTest
target_link_libraries(prac2_average_test PRIVATE gtest gtest_main)

# Adding the test(s) in CMake
add_test(NAME vector_addition COMMAND prac2_average_test)