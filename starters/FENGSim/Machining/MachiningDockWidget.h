#ifndef MACHININGDOCKER_H
#define MACHININGDOCKER_H

#include <QWidget>

namespace Ui {
class MachiningDockWidget;
}

class MachiningDockWidget : public QWidget
{
        Q_OBJECT

public:
        explicit MachiningDockWidget(QWidget *parent = 0);
        ~MachiningDockWidget();

public:
        Ui::MachiningDockWidget *ui;
};

#endif // MACHININGDOCKER_H
