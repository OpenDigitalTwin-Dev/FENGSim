#ifndef ROBOTTHREAD_H
#define ROBOTTHREAD_H

#include <QObject>
#include <QThread>
#include <QProcess>

class RobotThread : public QThread
{
public:
    RobotThread();
    void run () {
        QProcess *proc = new QProcess();
        proc->execute("bash", QStringList() << "-c" << "cd ../mbdyn/robot && ./run");
        quit();
        exec();
    }
};

#endif // ROBOTTHREAD_H

