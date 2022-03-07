include(utils)

function (add_3rd_party_library condition libName installDir)
    if (${${condition}})
        find_library_ex(${libName} LIB)
        message(STATUS "${libName} LIB_SONAME  = ${LIB_SONAME}")
        message(STATUS "${libName} LIB_ABSPATH = ${LIB_ABSPATH}")
        message(STATUS "${libName} LIB_LIBNAME = ${LIB_LIBNAME}")


        add_custom_target(make${libName}_symlink ALL
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIB_SONAME} ${libName}
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIB_LIBNAME} ${LIB_SONAME}
            BYPRODUCTS ${libName} ${LIB_SONAME} ${LIB_LIBNAME}
        )

        install(
            PROGRAMS
                "${LIB_ABSPATH}"
                "${CMAKE_CURRENT_BINARY_DIR}/${LIB_SONAME}"
                "${CMAKE_CURRENT_BINARY_DIR}/${libName}"
            DESTINATION ${installDir}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            COMPONENT Tests
        )
    endif()
endfunction()


add_3rd_party_library(ANY_SANITIZER "libstdc++.so" ${DCGM_TESTS_APP_DIR})
add_3rd_party_library(ADDRESS_SANITIZER "libasan.so" ${DCGM_TESTS_APP_DIR})
add_3rd_party_library(THREAD_SANITIZER "libtsan.so" ${DCGM_TESTS_APP_DIR})
add_3rd_party_library(LEAK_SANITIZER "liblsan.so" ${DCGM_TESTS_APP_DIR})