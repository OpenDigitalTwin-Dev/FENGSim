#ifndef RIVETDOCKWIDGET_H
#define RIVETDOCKWIDGET_H

#include <QWidget>

namespace Ui {
class RivetDockWidget;
}

class RivetDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RivetDockWidget(QWidget *parent = nullptr);
    ~RivetDockWidget();

public:
    Ui::RivetDockWidget *ui;
};

#endif // RIVETDOCKWIDGET_H
