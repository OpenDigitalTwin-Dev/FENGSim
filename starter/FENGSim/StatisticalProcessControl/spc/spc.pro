#-------------------------------------------------
#
# Project created by QtCreator 2019-12-10T20:28:45
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = spc
TEMPLATE = app
#TEMPLATE = lib

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        SPCMainWindow.cpp

HEADERS += \
        SPCMainWindow.h

FORMS += \
        SPCMainWindow.ui

QT += svg
INCLUDEPATH += /usr/local/lib/R/library/RInside/include/ /usr/local/lib/R/library/Rcpp/include/ /usr/local/lib/R/include
LIBS += -L/usr/local/lib/R/lib -lRlapack -lRblas -lR -L/usr/local/lib/R/library/RInside/lib/ -lRInside \
-L/usr/local/lib/R/library/Rcpp/libs/Rcpp.so

RESOURCES += \
    spcwind.qrc
