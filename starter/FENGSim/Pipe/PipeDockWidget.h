#ifndef PIPEDOCKWIDGET_H
#define PIPEDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class PipeDockWidget;
}

class PipeDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PipeDockWidget(QWidget *parent = nullptr);
    ~PipeDockWidget();

    Ui::PipeDockWidget *ui;
};

#endif // PIPEDOCKWIDGET_H
