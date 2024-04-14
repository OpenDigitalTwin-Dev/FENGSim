#ifndef OCPORODOCKWIDGET_H
#define OCPORODOCKWIDGET_H

#include <QWidget>

namespace Ui {
class OCPoroDockWidget;
}

class OCPoroDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OCPoroDockWidget(QWidget *parent = nullptr);
    ~OCPoroDockWidget();

public:
    Ui::OCPoroDockWidget *ui;
};

#endif // OCPORODOCKWIDGET_H
