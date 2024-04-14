#include "MeasureThread1.h"


void MeasureThread1::run () {
    //vtk_widget->SetSelectable(false);
    vtk_widget->MeasureImportSource(name, path);
    exit();
    exec();
}
