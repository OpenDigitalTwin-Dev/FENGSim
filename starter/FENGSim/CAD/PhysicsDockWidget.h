#ifndef PHYSICSDOCKWIDGET_H
#define PHYSICSDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class PhysicsDockWidget;
}

class PhysicsDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PhysicsDockWidget(QWidget *parent = nullptr);
    ~PhysicsDockWidget();

public:
    Ui::PhysicsDockWidget *ui;
};

#endif // PHYSICSDOCKWIDGET_H
