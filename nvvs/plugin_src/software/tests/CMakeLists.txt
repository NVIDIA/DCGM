set(CTEST_USE_LAUNCHERS 1)

find_package(Catch2 REQUIRED)
find_package(Yaml REQUIRED)
find_package(Jsoncpp REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)

    add_executable(softwaretests)
    target_sources(softwaretests
        PRIVATE
            ../Software.cpp
            SoftwareTests.cpp
            SoftwareTestsMain.cpp
    )
    
    target_include_directories(softwaretests PRIVATE ..)
    
    target_link_libraries(softwaretests
        PRIVATE
            nvvs_without_main_objects
            Catch2::Catch2
    )
    
    target_link_libraries(softwaretests PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS} rt dl)
    target_link_libraries(softwaretests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_link_options(softwaretests PRIVATE -rdynamic)

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(softwaretests EXTRA_ARGS --use-colour yes)
    endif()
endif()
