#ifndef MESHTHREAD1_H
#define MESHTHREAD1_H

#include <QThread>
#include <QProcess>
#include "Mesh/MeshGeneration.h"

class MeshThread1 : public QThread
{
    Q_OBJECT
public:
    MeshThread1();
    MeshModule* MM;
    TopoDS_Shape* S;
    double size;
    int refine_level;
    QString path;

    void run () {
        MM->MeshGeneration(S,size,refine_level,path);
    }
};

#endif // MESHTHREAD1_H
