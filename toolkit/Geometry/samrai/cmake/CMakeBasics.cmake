include(cmake/thirdparty/SetupSAMRAIThirdParty.cmake)

include(cmake/SetupM4.cmake)

include(cmake/SetupCompilers.cmake)

include(cmake/SAMRAIMacros.cmake)

# This needs to happen last, once all we have defined options and checked for
# third-party packages.
include(cmake/CMakeConfigureFile.cmake)
