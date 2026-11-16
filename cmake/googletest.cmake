include_guard(GLOBAL)

if(NOT ${RX_BUILD_TESTS})
    return()
endif()

include(FetchContent)

# GoogleTest
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)
target_include_directories(gtest BEFORE PUBLIC "${googletest_SOURCE_DIR}/googletest/include")

