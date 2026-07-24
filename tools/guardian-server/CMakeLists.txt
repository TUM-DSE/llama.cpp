# guardian-server: llama-server + Guardian shared-memory front-end.
# The shm protocol (header + implementation) is owned by the guardian repo
# and consumed from its native/ tree — single source of truth for the ABI.
set(TARGET llama-guardian-server)

set(GUARDIAN_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../native" CACHE PATH
    "Path to the guardian repo's native/ directory (canonical shm ABI)")

if (NOT EXISTS ${GUARDIAN_NATIVE_DIR}/include/guardian/shm.h)
    message(FATAL_ERROR
        "guardian-server requires the guardian repo's native/ tree; "
        "set -DGUARDIAN_NATIVE_DIR (looked at: ${GUARDIAN_NATIVE_DIR})")
endif()

add_executable(${TARGET}
    main.cpp
    guardian-shm.cpp
    ${GUARDIAN_NATIVE_DIR}/src/shm.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME guardian-server)
target_include_directories(${TARGET} PRIVATE
    ${GUARDIAN_NATIVE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../server)
target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)
