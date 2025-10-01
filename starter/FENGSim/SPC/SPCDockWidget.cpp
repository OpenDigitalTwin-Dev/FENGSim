#include "SPCDockWidget.h"
#include "ui_SPCDockWidget.h"
#include "SPCForm.h"

#include <QProcess>

SPCDockWidget::SPCDockWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SPCDockWidget)
{
    ui->setupUi(this);

    connect(ui->pushButton, SIGNAL(clicked(bool)), this, SLOT(openspcmodule()));
}

SPCDockWidget::~SPCDockWidget()
{
    delete ui;
}

void SPCDockWidget::check() {
    //if (ui->treeWidget->currentItem()->childCount() == 0) {
        //        QProcess *proc = new QProcess();
        //        proc->setWorkingDirectory( "/home/jiping/OpenDT/SPC/build-spc-Desktop_Qt_5_12_10_GCC_64bit-Debug" );
        //        proc->start(QString("./spc") );
        //SPCMainWindow w;
        //w.show();

        //        Form mm;
        //        mm.show();


    //}
}

void SPCDockWidget::openspcform() {
    //Dialog* mm = new Dialog();
    //Form* mm = new Form();
    SPCForm* w = new SPCForm();
    w->show();
}
