#ifndef ROBOTDOCKWIDGET_H
#define ROBOTDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class RobotDockWidget;
}

class RobotDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RobotDockWidget(QWidget *parent = nullptr);
    ~RobotDockWidget();


    Ui::RobotDockWidget *ui;
};

#endif // ROBOTDOCKWIDGET_H
