#ifndef MESHDOCKWIDGET_H
#define MESHDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class MeshDockWidget;
}

class MainWindow;

class MeshDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MeshDockWidget(QWidget *parent = 0);
    ~MeshDockWidget();
    friend class MainWindow;
private:
    Ui::MeshDockWidget *ui;
};

#endif // MESHDOCKWIDGET_H
