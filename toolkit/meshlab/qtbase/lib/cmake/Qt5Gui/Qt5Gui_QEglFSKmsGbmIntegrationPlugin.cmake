
add_library(Qt5::QEglFSKmsGbmIntegrationPlugin MODULE IMPORTED)


_populate_Gui_plugin_properties(QEglFSKmsGbmIntegrationPlugin RELEASE "egldeviceintegrations/libqeglfs-kms-integration.so" FALSE)

list(APPEND Qt5Gui_PLUGINS Qt5::QEglFSKmsGbmIntegrationPlugin)
set_property(TARGET Qt5::Gui APPEND PROPERTY QT_ALL_PLUGINS_egldeviceintegrations Qt5::QEglFSKmsGbmIntegrationPlugin)
set_property(TARGET Qt5::QEglFSKmsGbmIntegrationPlugin PROPERTY QT_PLUGIN_TYPE "egldeviceintegrations")
set_property(TARGET Qt5::QEglFSKmsGbmIntegrationPlugin PROPERTY QT_PLUGIN_EXTENDS "")
set_property(TARGET Qt5::QEglFSKmsGbmIntegrationPlugin PROPERTY QT_PLUGIN_CLASS_NAME "QEglFSKmsGbmIntegrationPlugin")
