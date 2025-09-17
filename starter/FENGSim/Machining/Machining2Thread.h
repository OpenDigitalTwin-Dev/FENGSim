#ifndef MACHINING2THREAD_H
#define MACHINING2THREAD_H

#include <QObject>
#include <QThread>
#include <QProcess>

class Machining2Thread : public QThread
{
public:
    Machining2Thread();
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

#endif // MACHINING2THREAD_H
