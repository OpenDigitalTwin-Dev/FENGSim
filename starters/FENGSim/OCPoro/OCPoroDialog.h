#ifndef OCPORODIALOG_H
#define OCPORODIALOG_H

#include <QDialog>

namespace Ui {
class OCPoroDialog;
}

class OCPoroDialog : public QDialog
{
    Q_OBJECT

public:
    explicit OCPoroDialog(QWidget *parent = nullptr);
    ~OCPoroDialog();

public:
    Ui::OCPoroDialog *ui;
};

#endif // OCPORODIALOG_H
