#ifndef SPCFORM_H
#define SPCFORM_H

#include <QWidget>
#include <QSvgWidget>
//#include "RInside.h"

namespace Ui {
class SPCForm;
}

class SPCForm : public QSvgWidget
{
    Q_OBJECT

public:
    explicit SPCForm(QWidget *parent = nullptr);
    ~SPCForm();

private:
    Ui::SPCForm *ui;

//private slots:
//    void getBandwidth(int bw);
//    void getKernel(int kernel);
//    void getRandomDataCmd(QString txt);
//    void runRandomDataCmd(void);

private:
    void setupDisplay(void);    // standard GUI boilderplate of arranging things
//    void plot(void);            // run a density plot in R and update the
//    void filterFile(void);      // modify the richer SVG produced by R

//    //QSvgWidget *m_svg;          // the SVG device
//    RInside m_R;              // reference to the R instance passed to constructor
//    QString m_tempfile;         // name of file used by R for plots
//    QString m_svgfile;          // another temp file, this time from Qt
//    int m_bw, m_kernel;         // parameters used to estimate the density
//    QString m_cmd;              // random draw command string
};

#endif // SPCFORM_H
