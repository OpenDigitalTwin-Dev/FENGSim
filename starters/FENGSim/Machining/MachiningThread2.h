#ifndef MACHININGTHREAD2_H
#define MACHININGTHREAD2_H



#include <QThread>
#include <QTimer>
#include <QDir>
#include <QFileInfoList>
#include <iostream>
#include <QProgressBar>

class MachiningThread2 : public QThread
{
    Q_OBJECT
public:
    MachiningThread2();
    QProgressBar* bar = new QProgressBar;
    int timestep_num = 5;
public slots:
    void vtknum()
    {
        QDir dir("/home/jiping/OpenDT/M++/data/vtk");
        QStringList stringlist_vtk;
        stringlist_vtk << "linear_elasticity_deform_tool_*.vtk";
        dir.setNameFilters(stringlist_vtk);
        QFileInfoList fileinfolist;
        fileinfolist = dir.entryInfoList();
        int files_num = fileinfolist.length();
        int proc = int(double(files_num) / double(timestep_num) * 100);
        std::cout << " vtk num: " << files_num << " " << timestep_num << " " << proc << std::endl;
        bar->setValue(proc);
        bar->update();
        if (files_num == timestep_num)
        {
            quit();
        }
    }
public:
    void run()
    {
        QTimer* timer = new QTimer;
        connect(timer, SIGNAL(timeout()), this, SLOT(vtknum()));
        timer->start(10);
        exec();
    }
};

#endif // MACHININGTHREAD2_H
