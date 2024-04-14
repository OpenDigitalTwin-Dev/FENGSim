#ifndef FEMWINDOW_H
#define FEMWINDOW_H

#include <QMainWindow>

#include "QVTKWidget.h"

namespace Ui {
class FEMWindow;
}

class FEMWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit FEMWindow(QWidget *parent = 0);
    ~FEMWindow();
    
public slots:
    void Solve ();
    void SolveDolfin ();

private:
    Ui::FEMWindow *ui;
    QVTKWidget* qvtk;
};

#endif // FEMWINDOW_H
