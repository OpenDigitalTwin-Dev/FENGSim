#-------------------------------------------------
#
# Project created by QtCreator 2017-02-03T12:40:26
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = FENGSim
TEMPLATE = app

#INCLUDEPATH += /usr/include/eigen3
#DISPLAY=:0
#LD_LIBRARY_PATH=/home/jiping/odt/Qt5.12.10/5.12.10/gcc_64/lib
#QT_PLUGIN_PATH=$$PWD/../../tools/qt/5.12.12/plugins
#QT_QPA_PLATFORM_PLUGIN_PATH=/home/jiping/odt/Qt5.12.10/5.12.10/gcc_64/plugins/platforms

SOURCES +=\
    CAD/PhysicsDockWidget.cpp \
    FEM/FEMThread1.cpp \
    Machining/Machining2Thread.cpp \
    Machining/MachiningDockWidget2.cpp \
    Main.cpp \
    MainWindow.cpp \
    CAD/OCCWidget.cpp \
    CAD/Primitive.cpp \
    CAD/CADDockWidget.cpp \
 #   Measure/MeasureThread1.cpp \
    Measure/MeasureICP.cpp \
    Measure/MeasureThread1.cpp \
    Measure/MeasureThread3.cpp \
    Mesh/MeshThread1.cpp \
    OCPoro/OCPoroDialog.cpp \
    OCPoro/OCPoroDockWidget.cpp \
#    Transport/data_analyze.cpp \
    Pipe/PipeDockWidget.cpp \
    Pipe/PipeThread.cpp \
    Rivet/RivetThread.cpp \
    Robot/RobotDockWidget.cpp \
    Robot/RobotThread.cpp \
    include/test.cpp \
    qcustomplot.cpp \
    AboutDialog.cpp \
    Mesh/MeshDockWidget.cpp \
    Mesh/MeshGeneration.cpp \
    Visual/VTKDockWidget.cpp \
    Visual/VTKWidget.cpp \
    FEM/FEMDockWidget.cpp \
    Measure/MeasureDockWidget.cpp \
    Measure/ls.cpp \
#    Measure/Registration.cpp \
#    Measure/example2.cpp \
    AdditiveManufacturing/AdditiveManufacturingDockWidget.cpp \
    Mesh/Slices.cpp \
    SPC/SPCDockWidget.cpp \
    SPC/SPCForm.cpp \
    AdditiveManufacturing/AMThread1.cpp \
    AdditiveManufacturing/AMThread2.cpp \
    AdditiveManufacturing/AMConfig.cpp \
    Machining/MachiningDockWidget.cpp \
    Machining/MachiningThread1.cpp \
    Machining/MachiningThread2.cpp \
    Machining/MakeTools.cpp \
    Transport/TransportDockWidget.cpp \
    Rivet/RivetDockWidget.cpp


HEADERS  += \
    CAD/PhysicsDockWidget.h \
    FEM/FEMThread1.h \
    Machining/Machining2Thread.h \
    Machining/MachiningDockWidget2.h \
    Machining/MakeTools.h \
    MainWindow.h \
    CAD/OCCWidget.h \
    CAD/Primitive.h \
    CAD/CADDockWidget.h \
#    Measure/MeasureThread1.h \
    Measure/MeasureICP.h \
    Measure/MeasureThread1.h \
    Measure/MeasureThread3.h \
    Mesh/MeshThread1.h \
    OCPoro/OCPoroDialog.h \
    OCPoro/OCPoroDockWidget.h \
#    Transport/data_analyze.h \
    Pipe/PipeDockWidget.h \
    Pipe/PipeThread.h \
    Rivet/RivetThread.h \
    Robot/RobotDockWidget.h \
    Robot/RobotThread.h \
    include/test.h \
    qcustomplot.h \
    AboutDialog.h \
    Mesh/MeshDockWidget.h \
    Mesh/MeshGeneration.h \
    Visual/VTKDockWidget.h \
    Visual/VTKWidget.h \
    vtkocc.h \
    FEM/FEMDockWidget.h \
    Measure/MeasureDockWidget.h \
    Measure/ls.h \
#    Measure/Registration.h \
#    Measure/example2.h \
    AdditiveManufacturing/AdditiveManufacturingDockWidget.h \
    Mesh/Slices.h \
    SPC/SPCDockWidget.h \
    SPC/SPCForm.h \
    AdditiveManufacturing/AMThread1.h \
    AdditiveManufacturing/AMThread2.h \
    AdditiveManufacturing/AMConfig.h \
    Machining/MachiningDockWidget.h \
    Machining/MachiningThread1.h \
    Machining/MachiningThread2.h \
    Transport/TransportDockWidget.h \
    Rivet/RivetDockWidget.h

FORMS    += \
    CAD/PhysicsDockWidget.ui \
    Machining/MachiningDockWidget2.ui \
    MainWindow.ui \
    AboutDialog.ui \
    CAD/CADDockWidget.ui \
    Mesh/MeshDockWidget.ui \
    OCPoro/OCPoroDialog.ui \
    OCPoro/OCPoroDockWidget.ui \
    Pipe/PipeDockWidget.ui \
    Robot/RobotDockWidget.ui \
    Visual/VTKDockWidget.ui \
    FEM/FEMDockWidget.ui \
    Measure/MeasureDockWidget.ui \
    AdditiveManufacturing/AdditiveManufacturingDockWidget.ui \
    SPC/SPCDockWidget.ui \
    SPC/SPCForm.ui \
    Machining/MachiningDockWidget.ui \
    Transport/TransportDockWidget.ui \
    Rivet/RivetDockWidget.ui

RESOURCES += \
    MainWindow.qrc \
    MeshWindow.qrc \
    FEMWindow.qrc \
    CADWindow.qrc \
    DataBaseWindow.qrc \
    MeasureWidnow.qrc \
    OCPoro.qrc \
    Pipe/fig/PipeWindow.qrc \
    Robot/RobotWindow.qrc \
    System.qrc \
    SPCWindow.qrc \
    AMWindow.qrc \
    Machining/machining.qrc \
    TransportWindow.qrc \
    Rivet/RivetWindow.qrc

INCLUDEPATH += $$PWD/../../toolkit/Geometry/install/oce_install/include/oce $$PWD/../../toolkit/Geometry/install/gmsh_install/include/gmsh $$PWD/../../toolkit/Geometry/install/vtk_install/include/vtk-8.1

LIBS += -L$$PWD/../../toolkit/Geometry/install/oce_install/lib \
-lTKBinL     -lTKBO     -lTKFeat      -lTKGeomBase  -lTKMath    -lTKOpenGl  -lTKService      -lTKSTEP209   -lTKTObj     -lTKXCAFSchema  -lTKXmlL \
-lTKBin      -lTKBRep   -lTKFillet    -lTKHLR       -lTKMesh    -lTKPCAF    -lTKShapeSchema  -lTKSTEPAttr  -lTKTopAlgo  -lTKXCAF        -lTKXml \
-lTKBinTObj  -lTKCAF    -lTKG2d       -lTKIGES      -lTKMeshVS  -lTKPLCAF   -lTKShHealing    -lTKSTEPBase  -lTKV3d      -lTKXDEIGES     -lTKXmlTObj \
-lTKBinXCAF  -lTKCDF    -lTKG3d       -lTKIVtk      -lTKNIS     -lTKPrim    -lTKStdLSchema   -lTKSTEP      -lTKVoxel    -lTKXDESTEP     -lTKXmlXCAF \
-lTKBool     -lTKernel  -lTKGeomAlgo  -lTKLCAF      -lTKOffset  -lTKPShape  -lTKStdSchema    -lTKSTL       -lTKVRML     -lTKXMesh       -lTKXSBase
LIBS += -L$$PWD/../../toolkit/Geometry/install/gmsh_install/lib -lGmsh
LIBS += -L$$PWD/../../toolkit/Geometry/install/vtk_install/lib \
-lvtkalglib-8.1                       -lvtkFiltersHyperTree-8.1        -lvtkImagingFourier-8.1        -lvtkIOLSDyna-8.1        -lvtkRenderingAnnotation-8.1 \
-lvtkChartsCore-8.1                   -lvtkFiltersImaging-8.1          -lvtkImagingGeneral-8.1        -lvtkIOMINC-8.1          -lvtkRenderingContext2D-8.1 \
-lvtkCommonColor-8.1                  -lvtkFiltersModeling-8.1         -lvtkImagingHybrid-8.1         -lvtkIOMovie-8.1         -lvtkRenderingContextOpenGL2-8.1 \
-lvtkCommonComputationalGeometry-8.1  -lvtkFiltersParallel-8.1         -lvtkImagingMath-8.1           -lvtkIONetCDF-8.1        -lvtkRenderingCore-8.1 \
-lvtkCommonCore-8.1                   -lvtkFiltersParallelImaging-8.1  -lvtkImagingMorphological-8.1  -lvtkIOParallel-8.1      -lvtkRenderingFreeType-8.1 \
-lvtkCommonDataModel-8.1              -lvtkFiltersPoints-8.1           -lvtkImagingSources-8.1        -lvtkIOParallelXML-8.1   -lvtkRenderingGL2PSOpenGL2-8.1 \
-lvtkCommonExecutionModel-8.1         -lvtkFiltersProgrammable-8.1     -lvtkImagingStatistics-8.1     -lvtkIOPLY-8.1           -lvtkRenderingImage-8.1 \
-lvtkCommonMath-8.1                   -lvtkFiltersSelection-8.1        -lvtkImagingStencil-8.1        -lvtkIOSQL-8.1           -lvtkRenderingLabel-8.1 \
-lvtkCommonMisc-8.1                   -lvtkFiltersSMP-8.1              -lvtkInfovisCore-8.1           -lvtkIOTecplotTable-8.1  -lvtkRenderingLOD-8.1 \
-lvtkCommonSystem-8.1                 -lvtkFiltersSources-8.1          -lvtkInfovisLayout-8.1         -lvtkIOVideo-8.1         -lvtkRenderingOpenGL2-8.1 \
-lvtkCommonTransforms-8.1             -lvtkFiltersStatistics-8.1       -lvtkInteractionImage-8.1      -lvtkIOXML-8.1           -lvtkRenderingQt-8.1 \
-lvtkDICOMParser-8.1                  -lvtkFiltersTexture-8.1          -lvtkInteractionStyle-8.1      -lvtkIOXMLParser-8.1     -lvtkRenderingVolume-8.1 \
-lvtkDomainsChemistry-8.1             -lvtkFiltersTopology-8.1         -lvtkInteractionWidgets-8.1    -lvtkjpeg-8.1            -lvtkRenderingVolumeOpenGL2-8.1 \
-lvtkDomainsChemistryOpenGL2-8.1      -lvtkFiltersVerdict-8.1          -lvtkIOAMR-8.1                 -lvtkjsoncpp-8.1         -lvtksqlite-8.1 \
-lvtkexoIIc-8.1                       -lvtkfreetype-8.1                -lvtkIOCore-8.1                -lvtklibharu-8.1         -lvtksys-8.1 \
-lvtkexpat-8.1                        -lvtkGeovisCore-8.1              -lvtkIOEnSight-8.1             -lvtklibxml2-8.1         -lvtktiff-8.1 \
-lvtkFiltersAMR-8.1                   -lvtkgl2ps-8.1                   -lvtkIOExodus-8.1              -lvtklz4-8.1             -lvtkverdict-8.1 \
-lvtkFiltersCore-8.1                  -lvtkglew-8.1                    -lvtkIOExport-8.1              -lvtkmetaio-8.1          -lvtkViewsContext2D-8.1 \
-lvtkFiltersExtraction-8.1            -lvtkGUISupportQt-8.1            -lvtkIOExportOpenGL2-8.1       -lvtkNetCDF-8.1          -lvtkViewsCore-8.1 \
-lvtkFiltersFlowPaths-8.1             -lvtkGUISupportQtSQL-8.1         -lvtkIOGeometry-8.1            -lvtknetcdfcpp-8.1       -lvtkViewsInfovis-8.1 \
-lvtkFiltersGeneral-8.1               -lvtkhdf5-8.1                    -lvtkIOImage-8.1               -lvtkoggtheora-8.1       -lvtkViewsQt-8.1 \
-lvtkFiltersGeneric-8.1               -lvtkhdf5_hl-8.1                 -lvtkIOImport-8.1              -lvtkParallelCore-8.1    -lvtkzlib-8.1 \
-lvtkFiltersGeometry-8.1              -lvtkImagingColor-8.1            -lvtkIOInfovis-8.1             -lvtkpng-8.1 \
-lvtkFiltersHybrid-8.1                -lvtkImagingCore-8.1             -lvtkIOLegacy-8.1              -lvtkproj4-8.1
#-lG4analysis           -lG4global         -lG4particles     -lG4Tree \
#-lG4clhep              -lG4GMocren        -lG4persistency   -lG4visHepRep \
#-lG4digits_hits        -lG4graphics_reps  -lG4physicslists  -lG4vis_management \
#-lG4error_propagation  -lG4intercoms      -lG4processes     -lG4visXXX \
#-lG4event              -lG4interfaces     -lG4RayTracer     -lG4VRML \
#-lG4expat              -lG4materials      -lG4readout       -lG4zlib \
#-lG4FR                 -lG4modeling       -lG4run \
#-lG4geometry           -lG4OpenGL         -lG4tracking \
#-lG4gl2ps              -lG4parmodels      -lG4track \
#-L/home/jiping/FENGSim/FENGSim/GDT/build -lexample2

QT += svg

# no necessary use RInside, since have to add more libs, the spc just generate a svg and use qsvg import
# so change to the staters

#INCLUDEPATH += /usr/local/lib/R/library/RInside/include/ /usr/local/lib/R/library/Rcpp/include/ /usr/local/lib/R/include
#LIBS += -L/usr/local/lib/R/lib -lRlapack -lRblas -lR -L/usr/local/lib/R/library/RInside/lib/ -lRInside \
#-L/usr/local/lib/R/library/Rcpp/libs/Rcpp.so

#INCLUDEPATH += $$PWD/../../toolkit/PS/install/r_install/lib/R/library/RInside/include/ \
#               $$PWD/../../toolkit/PS/install/r_install/lib/R/library/Rcpp/include/ \
#               $$PWD/../../toolkit/PS/install/r_install/lib/R/include \
#               $$PWD/../../toolkit/PS/install/r_install/lib/R/include
#LIBS += -L/$$PWD/../../toolkit/PS/install/r_install/lib/R/lib -lRlapack -lRblas -lR \
#        -L/$$PWD/../../toolkit/PS/install/r_install/lib/R/library/Rcpp/libs/Rcpp.so \
#        -L/$$PWD/../../toolkit/PS/install/r_install/lib/R/library/RInside/lib/ -lRInside


#QMAKE_LFLAGS += -no-pie

DISTFILES += \
    figure/cad_wind/tree.png

