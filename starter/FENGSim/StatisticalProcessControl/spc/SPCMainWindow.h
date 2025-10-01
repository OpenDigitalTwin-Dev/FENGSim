#ifndef SPCMAINWINDOW_H
#define SPCMAINWINDOW_H

#include <QMainWindow>

#include <RInside.h>

#include <QtGui>
#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QRadioButton>
#include <QGroupBox>
#include <QButtonGroup>
#include <QMainWindow>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include <QTemporaryFile>
#include <QSvgWidget>

namespace Ui {
class SPCMainWindow;
}

class SPCMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit SPCMainWindow(QWidget *parent = 0);
    ~SPCMainWindow();

private:
    Ui::SPCMainWindow *ui;

private slots:
    void getBandwidth(int bw);
    void getKernel(int kernel);
    void getRandomDataCmd(QString txt);
    void runRandomDataCmd(void);

private:
    void setupDisplay(void);    // standard GUI boilderplate of arranging things
    void plot(void);            // run a density plot in R and update the
    void filterFile(void);      // modify the richer SVG produced by R

    QSvgWidget *m_svg;          // the SVG device
    RInside m_R;              // reference to the R instance passed to constructor
    QString m_tempfile;         // name of file used by R for plots
    QString m_svgfile;          // another temp file, this time from Qt
    int m_bw, m_kernel;         // parameters used to estimate the density
    QString m_cmd;              // random draw command string
};

#endif // SPCMAINWINDOW_H
