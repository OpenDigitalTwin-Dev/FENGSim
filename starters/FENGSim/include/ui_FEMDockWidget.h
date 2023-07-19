/********************************************************************************
** Form generated from reading UI file 'FEMDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FEMDOCKWIDGET_H
#define UI_FEMDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FEMDockWidget
{
public:
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout_2;
    QLabel *label_4;
    QComboBox *comboBox_4;
    QPushButton *pushButton_2;
    QComboBox *comboBox;
    QPushButton *pushButton;
    QLabel *label;
    QGridLayout *gridLayout;
    QComboBox *comboBox_2;
    QComboBox *comboBox_3;
    QLineEdit *lineEdit;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *FEMDockWidget)
    {
        if (FEMDockWidget->objectName().isEmpty())
            FEMDockWidget->setObjectName(QString::fromUtf8("FEMDockWidget"));
        FEMDockWidget->resize(213, 400);
        FEMDockWidget->setMaximumSize(QSize(213, 400));
        verticalLayout = new QVBoxLayout(FEMDockWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_4 = new QLabel(FEMDockWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        QFont font;
        font.setPointSize(9);
        label_4->setFont(font);

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        comboBox_4 = new QComboBox(FEMDockWidget);
        comboBox_4->setObjectName(QString::fromUtf8("comboBox_4"));
        comboBox_4->setMinimumSize(QSize(0, 25));
        comboBox_4->setMaximumSize(QSize(16777215, 25));
        comboBox_4->setFont(font);

        gridLayout_2->addWidget(comboBox_4, 1, 1, 1, 1);

        pushButton_2 = new QPushButton(FEMDockWidget);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setMinimumSize(QSize(25, 25));
        pushButton_2->setMaximumSize(QSize(25, 25));
        pushButton_2->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"    padding: 2px;\n"
"}"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_2->setIcon(icon);
        pushButton_2->setCheckable(false);

        gridLayout_2->addWidget(pushButton_2, 1, 2, 1, 1);

        comboBox = new QComboBox(FEMDockWidget);
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setEnabled(true);
        comboBox->setMinimumSize(QSize(0, 25));
        comboBox->setMaximumSize(QSize(16777215, 25));
        comboBox->setFont(font);

        gridLayout_2->addWidget(comboBox, 0, 1, 1, 1);

        pushButton = new QPushButton(FEMDockWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setEnabled(true);
        pushButton->setMinimumSize(QSize(25, 25));
        pushButton->setMaximumSize(QSize(25, 25));
        QFont font1;
        font1.setPointSize(10);
        pushButton->setFont(font1);
        pushButton->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"    padding: 2px;\n"
"}"));
        pushButton->setIcon(icon);

        gridLayout_2->addWidget(pushButton, 0, 2, 1, 1);

        label = new QLabel(FEMDockWidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFont(font);

        gridLayout_2->addWidget(label, 0, 0, 1, 1);


        verticalLayout->addLayout(gridLayout_2);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        comboBox_2 = new QComboBox(FEMDockWidget);
        comboBox_2->setObjectName(QString::fromUtf8("comboBox_2"));
        comboBox_2->setEnabled(false);
        comboBox_2->setMinimumSize(QSize(0, 25));
        comboBox_2->setMaximumSize(QSize(16777215, 25));
        comboBox_2->setFont(font);

        gridLayout->addWidget(comboBox_2, 1, 0, 1, 1);

        comboBox_3 = new QComboBox(FEMDockWidget);
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->setObjectName(QString::fromUtf8("comboBox_3"));
        comboBox_3->setEnabled(false);
        comboBox_3->setMinimumSize(QSize(0, 25));
        comboBox_3->setMaximumSize(QSize(16777215, 25));
        comboBox_3->setFont(font);

        gridLayout->addWidget(comboBox_3, 2, 0, 1, 1);

        lineEdit = new QLineEdit(FEMDockWidget);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        lineEdit->setEnabled(false);
        lineEdit->setMinimumSize(QSize(0, 25));
        lineEdit->setMaximumSize(QSize(16777215, 25));
        lineEdit->setFont(font);

        gridLayout->addWidget(lineEdit, 3, 0, 1, 1);


        verticalLayout->addLayout(gridLayout);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(FEMDockWidget);

        QMetaObject::connectSlotsByName(FEMDockWidget);
    } // setupUi

    void retranslateUi(QWidget *FEMDockWidget)
    {
        FEMDockWidget->setWindowTitle(QApplication::translate("FEMDockWidget", "Form", nullptr));
        label_4->setText(QApplication::translate("FEMDockWidget", "Examples", nullptr));
        pushButton_2->setText(QString());
        pushButton->setText(QString());
        label->setText(QApplication::translate("FEMDockWidget", "Equation", nullptr));
        comboBox_3->setItemText(0, QApplication::translate("FEMDockWidget", "0", nullptr));
        comboBox_3->setItemText(1, QApplication::translate("FEMDockWidget", "1", nullptr));
        comboBox_3->setItemText(2, QApplication::translate("FEMDockWidget", "2", nullptr));
        comboBox_3->setItemText(3, QApplication::translate("FEMDockWidget", "3", nullptr));
        comboBox_3->setItemText(4, QApplication::translate("FEMDockWidget", "4", nullptr));
        comboBox_3->setItemText(5, QApplication::translate("FEMDockWidget", "5", nullptr));
        comboBox_3->setItemText(6, QApplication::translate("FEMDockWidget", "6", nullptr));
        comboBox_3->setItemText(7, QApplication::translate("FEMDockWidget", "7", nullptr));

    } // retranslateUi

};

namespace Ui {
    class FEMDockWidget: public Ui_FEMDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FEMDOCKWIDGET_H
