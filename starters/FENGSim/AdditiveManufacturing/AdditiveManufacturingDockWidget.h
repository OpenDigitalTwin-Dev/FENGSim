#ifndef ADDITIVEMANUFACTURINGDOCKWIDGET_H
#define ADDITIVEMANUFACTURINGDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class AdditiveManufacturingDockWidget;
}

class AdditiveManufacturingDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit AdditiveManufacturingDockWidget(QWidget *parent = 0);
    ~AdditiveManufacturingDockWidget();

public:
    Ui::AdditiveManufacturingDockWidget *ui;
};

#endif // ADDITIVEMANUFACTURINGDOCKWIDGET_H
