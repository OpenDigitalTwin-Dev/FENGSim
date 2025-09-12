#ifndef PIPETHREAD_H
#define PIPETHREAD_H

#include <QObject>
#include <QThread>
#include <QProcess>

class PipeThread : public QThread
{
public:
    PipeThread();
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

#endif // PIPETHREAD_H
