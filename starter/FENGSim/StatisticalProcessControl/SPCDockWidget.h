#ifndef SPCDOCKWIDGET_H
#define SPCDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class SPCDockWidget;
}

class SPCDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SPCDockWidget(QWidget *parent = 0);
    ~SPCDockWidget();

public slots:
    void check ();

private:
    Ui::SPCDockWidget *ui;
};

#endif // SPCDOCKWIDGET_H
