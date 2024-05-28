#ifndef FEMTHREAD1_H
#define FEMTHREAD1_H

#include <QThread>
#include <QProcess>

class FEMThread1 : public QThread
{
    Q_OBJECT
public:
    FEMThread1();
    void run () {
        QProcess *proc = new QProcess();
        proc->setWorkingDirectory( "../Elasticity/build" );
        QString command(QString("mpirun -np 4 ./ElasticitySolver"));
        proc->start(command);
        if (proc->waitForFinished(-1)) {
            quit();
        }
        exec();
    }
};

#endif // FEMTHREAD1_H
