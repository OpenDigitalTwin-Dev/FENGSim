#ifndef VISUALDOCKWIDGET_H
#define VISUALDOCKWIDGET_H

#include <QWidget>
#include <QMenu>

namespace Ui {
class VTKDockWidget;
}

class MainWindow;

class VTKDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit VTKDockWidget(QWidget *parent = 0);
    ~VTKDockWidget();
    friend class MainWindow;
private:
    Ui::VTKDockWidget *ui;
};


#endif // VISUALDOCKWIDGET_H
