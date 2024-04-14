#ifndef ABOUTDIALOG_H
#define ABOUTDIALOG_H

#include <QDialog>
#include <QLabel>

namespace Ui {
class AboutDialog;
}

class AboutDialog : public QDialog
{
        Q_OBJECT

public:
        explicit AboutDialog(QWidget *parent = 0);
        ~AboutDialog();
        void ChangePicture (QString name);


public slots:
        // *******************************************************************
        //
        // close dialog
        //
        // *******************************************************************
        void CloseDialog ()
        {
                this->close();
        }
private:
        Ui::AboutDialog *ui;
};
// *******************************************************************
//
// define a new label class to send click signal
//
// *******************************************************************
class ClickedLabel : public QLabel
{
        Q_OBJECT
public:
        explicit ClickedLabel(QWidget* parent=0 );
        ~ClickedLabel();
signals:
        // *******************************************************************
        // define clicked signal
        // *******************************************************************
        void clicked();
protected:
        // *******************************************************************
        // redefine mouse click
        // *******************************************************************
        void mousePressEvent(QMouseEvent* event);


};
#endif // ABOUTDIALOG_H
