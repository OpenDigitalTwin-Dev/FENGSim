#ifndef AMTHREAD2_H
#define AMTHREAD2_H

#include <QThread>
#include <QTimer>
#include <QDir>
#include <QFileInfoList>
#include <iostream>
#include <QProgressBar>

class AMThread2 : public QThread
{
        Q_OBJECT
public:
        AMThread2();
        QProgressBar* bar = new QProgressBar;
        int timestep_num = 101;
public slots:
        void vtknum()
        {
                QDir dir("./../../AM/data/vtk");
                QStringList stringlist_vtk;
                stringlist_vtk << "am_mesh_*.vtk";
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

#endif // AMTHREAD2_H
