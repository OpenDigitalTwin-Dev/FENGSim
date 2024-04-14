/********************************************************************************
** Form generated from reading UI file 'VTKDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VTKDOCKWIDGET_H
#define UI_VTKDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VTKDockWidget
{
public:
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *lineEdit;
    QPushButton *pushButton;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QPushButton *pushButton_2;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *VTKDockWidget)
    {
        if (VTKDockWidget->objectName().isEmpty())
            VTKDockWidget->setObjectName(QString::fromUtf8("VTKDockWidget"));
        VTKDockWidget->resize(213, 400);
        VTKDockWidget->setMaximumSize(QSize(213, 400));
        verticalLayout = new QVBoxLayout(VTKDockWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(VTKDockWidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setMinimumSize(QSize(0, 25));
        label->setMaximumSize(QSize(16777215, 25));
        QFont font;
        font.setPointSize(9);
        label->setFont(font);

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineEdit = new QLineEdit(VTKDockWidget);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(lineEdit->sizePolicy().hasHeightForWidth());
        lineEdit->setSizePolicy(sizePolicy);
        lineEdit->setMinimumSize(QSize(90, 25));
        lineEdit->setMaximumSize(QSize(90, 25));

        gridLayout->addWidget(lineEdit, 0, 1, 1, 1);

        pushButton = new QPushButton(VTKDockWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
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
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton->setIcon(icon);

        gridLayout->addWidget(pushButton, 0, 2, 1, 1);


        verticalLayout->addLayout(gridLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(VTKDockWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);

        horizontalLayout->addWidget(label_2);

        pushButton_2 = new QPushButton(VTKDockWidget);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setMinimumSize(QSize(25, 25));
        pushButton_2->setMaximumSize(QSize(25, 25));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/amwind/figure/am_wind/open.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_2->setIcon(icon1);

        horizontalLayout->addWidget(pushButton_2);


        verticalLayout_2->addLayout(horizontalLayout);


        verticalLayout->addLayout(verticalLayout_2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(VTKDockWidget);

        QMetaObject::connectSlotsByName(VTKDockWidget);
    } // setupUi

    void retranslateUi(QWidget *VTKDockWidget)
    {
        VTKDockWidget->setWindowTitle(QApplication::translate("VTKDockWidget", "Form", nullptr));
        label->setText(QApplication::translate("VTKDockWidget", "VTK: ", nullptr));
        pushButton->setText(QString());
        label_2->setText(QApplication::translate("VTKDockWidget", "Experiment Data:", nullptr));
        pushButton_2->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class VTKDockWidget: public Ui_VTKDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VTKDOCKWIDGET_H
