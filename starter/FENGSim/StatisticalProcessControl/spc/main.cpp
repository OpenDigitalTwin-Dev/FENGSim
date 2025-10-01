#include "SPCMainWindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SPCMainWindow w;
    w.show();

    return a.exec();
}
