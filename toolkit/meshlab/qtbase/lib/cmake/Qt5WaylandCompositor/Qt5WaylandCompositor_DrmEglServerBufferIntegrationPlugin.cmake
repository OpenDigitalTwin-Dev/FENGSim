
add_library(Qt5::DrmEglServerBufferIntegrationPlugin MODULE IMPORTED)


_populate_WaylandCompositor_plugin_properties(DrmEglServerBufferIntegrationPlugin RELEASE "wayland-graphics-integration-server/libqt-wayland-compositor-drm-egl-server-buffer.so" FALSE)

list(APPEND Qt5WaylandCompositor_PLUGINS Qt5::DrmEglServerBufferIntegrationPlugin)
set_property(TARGET Qt5::WaylandCompositor APPEND PROPERTY QT_ALL_PLUGINS_wayland_graphics_integration_server Qt5::DrmEglServerBufferIntegrationPlugin)
set_property(TARGET Qt5::DrmEglServerBufferIntegrationPlugin PROPERTY QT_PLUGIN_TYPE "wayland-graphics-integration-server")
set_property(TARGET Qt5::DrmEglServerBufferIntegrationPlugin PROPERTY QT_PLUGIN_EXTENDS "")
set_property(TARGET Qt5::DrmEglServerBufferIntegrationPlugin PROPERTY QT_PLUGIN_CLASS_NAME "DrmEglServerBufferIntegrationPlugin")
