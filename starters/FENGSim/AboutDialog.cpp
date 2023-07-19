#include "AboutDialog.h"
#include "ui_AboutDialog.h"
#include <QFile>

AboutDialog::AboutDialog(QWidget *parent) :
        QDialog(parent),
        ui(new Ui::AboutDialog)
{
        ui->setupUi(this);
        //        QFile qss(":/main_wind/style.qss");
        //        if(qss.open(QFile::ReadOnly)){
        //                qApp->setStyleSheet(qss.readAll());
        //                qss.close();
        //        }
        // delete the title of the dialog
        setWindowFlags(Qt::FramelessWindowHint);
        // click the label and close the dialog
        connect(ui->label,SIGNAL(clicked()),this,SLOT(CloseDialog()));


}
AboutDialog::~AboutDialog()
{
        delete ui;
}
void AboutDialog::ChangePicture(QString name)
{
        //  ui->label->setPixmap(QPixmap(name));
}
ClickedLabel::ClickedLabel(QWidget* parent)
        : QLabel(parent)
{
}
ClickedLabel::~ClickedLabel()
{
}
void ClickedLabel::mousePressEvent(QMouseEvent* event)
{
        emit clicked();
}
