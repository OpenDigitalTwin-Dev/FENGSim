
add_library(Qt5::QSdl2GamepadBackendPlugin MODULE IMPORTED)


_populate_Gamepad_plugin_properties(QSdl2GamepadBackendPlugin RELEASE "gamepads/libsdl2gamepad.so" FALSE)

list(APPEND Qt5Gamepad_PLUGINS Qt5::QSdl2GamepadBackendPlugin)
set_property(TARGET Qt5::Gamepad APPEND PROPERTY QT_ALL_PLUGINS_gamepads Qt5::QSdl2GamepadBackendPlugin)
set_property(TARGET Qt5::QSdl2GamepadBackendPlugin PROPERTY QT_PLUGIN_TYPE "gamepads")
set_property(TARGET Qt5::QSdl2GamepadBackendPlugin PROPERTY QT_PLUGIN_EXTENDS "")
set_property(TARGET Qt5::QSdl2GamepadBackendPlugin PROPERTY QT_PLUGIN_CLASS_NAME "QSdl2GamepadBackendPlugin")
