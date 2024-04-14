#ifndef FEMDOCKWIDGET_H
#define FEMDOCKWIDGET_H

#include <QWidget>
#include <QComboBox>

namespace Ui {
class FEMDockWidget;
}

class MainWindow;

class FEMDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FEMDockWidget(QWidget *parent = 0);
    ~FEMDockWidget();
    friend class MainWindow;

public slots:
    void MainModule ();
    void Configure ();
    void OpenMeshFile ();

private:
    Ui::FEMDockWidget *ui;
    QString mesh_file;
};

#endif // FEMDOCKWIDGET_H
