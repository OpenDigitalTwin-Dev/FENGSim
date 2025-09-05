#ifndef RIVETTHREAD_H
#define RIVETTHREAD_H

#include <QThread>
#include <QProcess>

class RivetThread : public QThread
{
    Q_OBJECT
public:
    RivetThread();
    void run () {
        QProcess *proc = new QProcess();
        proc->setWorkingDirectory( "./../../toolkit/MultiX/build" );
        proc->start("./multix");
        if (proc->waitForFinished(-1)) {
            quit();
        }
        exec();
    }
};

#endif // RIVETTHREAD_H
