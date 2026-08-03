set(TARGET llama-rag-scheduler)

add_library(${TARGET} STATIC rag-scheduler.cpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
