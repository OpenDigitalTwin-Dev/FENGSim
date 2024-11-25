
add_library(Qt5::ShmServerBufferIntegrationPlugin MODULE IMPORTED)


_populate_WaylandCompositor_plugin_properties(ShmServerBufferIntegrationPlugin RELEASE "wayland-graphics-integration-server/libqt-wayland-compositor-shm-emulation-server.so" FALSE)

list(APPEND Qt5WaylandCompositor_PLUGINS Qt5::ShmServerBufferIntegrationPlugin)
set_property(TARGET Qt5::WaylandCompositor APPEND PROPERTY QT_ALL_PLUGINS_wayland_graphics_integration_server Qt5::ShmServerBufferIntegrationPlugin)
set_property(TARGET Qt5::ShmServerBufferIntegrationPlugin PROPERTY QT_PLUGIN_TYPE "wayland-graphics-integration-server")
set_property(TARGET Qt5::ShmServerBufferIntegrationPlugin PROPERTY QT_PLUGIN_EXTENDS "")
set_property(TARGET Qt5::ShmServerBufferIntegrationPlugin PROPERTY QT_PLUGIN_CLASS_NAME "ShmServerBufferIntegrationPlugin")
