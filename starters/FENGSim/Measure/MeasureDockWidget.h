#ifndef MEASUREDOCKWIDGET_H
#define MEASUREDOCKWIDGET_H

#include <QWidget>
#include <QMenu>

namespace Ui {
class MeasureDockWidget;
}

enum MeasureObject {measure_surface, measure_line};
enum MeasureType {straightness, flatness, circularity, cylindricity, lineprofile, surfaceprofile};

class MeasureDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MeasureDockWidget(QWidget *parent = 0);
    ~MeasureDockWidget();

public:
    Ui::MeasureDockWidget *ui;
    QMenu* menu_objects;
    QMenu* menu_cdt;
    MeasureObject measure_obj;
    MeasureType measure_type;
    QMenu* gdt_objects_type;

public slots:
    void SetObject1 ();
    void SetObject2 ();
    void SetAllObjectsUnchecked ();
    void SetType1 ();
    void SetType2 ();
    void SetType3 ();
    void SetType4 ();
    void SetType5 ();
    void SetType6 ();
    void SetAllTypesUnchecked ();


    void SetSingleStep ();

};

#endif // MEASUREDOCKWIDGET_H
