# CMake minimal requiered version
cmake_minimum_required(VERSION 3.18)

# TP3
project(TP3 LANGUAGES CXX CUDA)

# Fichier à executer
set(EXECUTABLES reduction reduction_test reduction_adaptative reduction_multiblock)


set(SRC_FOLDER "src") # Répertoire source
set(EXT_FILE ".cu") # Extention du fichier à compiler (.c, .cpp, .cu, etc)


# Librairies (ajout des librairies à utiliser pour l'exécution) ex : set(TARGET_LIB curand)
set(TARGET_LIB )


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
list(FILTER SOURCE_FILES EXCLUDE REGEX "${SRC_FOLDER}/[${EXECUTABLES}]+\\${EXT_FILE}$")

# Boucle pour créer chaque exécutable
foreach(EXE ${EXECUTABLES})
    add_executable(${EXE} ${SRC_FOLDER}/${EXE}${EXT_FILE} ${SRC_FOLDER}/${EXE}${EXT_FILE} ${SOURCE_FILES})
    set_target_properties(${EXE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${EXE} PRIVATE ${TARGET_LIB})
endforeach()


################
#Unitary tests
###############

enable_testing()


add_subdirectory(/home/elcalderonymora/googletest ${CMAKE_BINARY_DIR}/googletest_build)


# Link with GoogleTest
target_link_libraries(reduction_test PRIVATE gtest gtest_main)

# Adding the test(s) in CMake
add_test(NAME vector_addition COMMAND reduction_test)