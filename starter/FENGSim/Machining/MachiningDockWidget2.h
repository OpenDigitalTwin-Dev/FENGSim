#ifndef MACHININGDOCKWIDGET2_H
#define MACHININGDOCKWIDGET2_H

#include <QWidget>

namespace Ui {
class MachiningDockWidget2;
}

class MachiningDockWidget2 : public QWidget
{
    Q_OBJECT

public:
    explicit MachiningDockWidget2(QWidget *parent = nullptr);
    ~MachiningDockWidget2();

    Ui::MachiningDockWidget2 *ui;
};

#endif // PIPEDOCKWIDGET_H
