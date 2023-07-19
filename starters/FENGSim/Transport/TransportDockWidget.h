#ifndef TRANSPORTDOCKWIDGET_H
#define TRANSPORTDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class TransportDockWidget;
}

class TransportDockWidget : public QWidget
{
        Q_OBJECT

public:
        explicit TransportDockWidget(QWidget *parent = 0);
        ~TransportDockWidget();


        Ui::TransportDockWidget *ui;
};

#endif // TRANSPORTDOCKWIDGET_H
