#ifndef MACHININGTHREAD1_H
#define MACHININGTHREAD1_H



#include <QThread>
#include <QProcess>

class MachiningThread1 : public QThread
{
        Q_OBJECT
public:
        MachiningThread1();
        void run () {
                QProcess *proc = new QProcess();
                proc->setWorkingDirectory( "/home/jiping/OpenDT/M++/" );
                QString command(QString("./MachiningRun"));
                proc->start(command);
                if (proc->waitForFinished(-1)) {
                        quit();
                }
                exec();
        }
};

#endif // MACHININGTHREAD1_H
