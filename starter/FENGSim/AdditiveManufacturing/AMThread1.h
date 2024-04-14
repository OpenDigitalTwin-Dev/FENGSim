#ifndef AMTHREAD1_H
#define AMTHREAD1_H

#include <QThread>
#include <QProcess>

class AMThread1 : public QThread
{
        Q_OBJECT
public:
        AMThread1();
        void run () {
                QProcess *proc = new QProcess();
                proc->setWorkingDirectory( "./../../AM" );
                QString command(QString("./AMRun"));
                proc->start(command);
                if (proc->waitForFinished(-1)) {
                        quit();
                }
                exec();
        }
};

#endif // AMTHREAD1_H
