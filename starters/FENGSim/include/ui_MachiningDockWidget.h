/********************************************************************************
** Form generated from reading UI file 'MachiningDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MACHININGDOCKWIDGET_H
#define UI_MACHININGDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MachiningDockWidget
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_5;
    QComboBox *comboBox_5;
    QPushButton *pushButton_7;
    QGridLayout *gridLayout;
    QPushButton *pushButton;
    QLineEdit *lineEdit;
    QComboBox *comboBox;
    QComboBox *comboBox_2;
    QLineEdit *lineEdit_2;
    QPushButton *pushButton_2;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_3;
    QSpacerItem *horizontalSpacer_4;
    QPushButton *pushButton_11;
    QHBoxLayout *horizontalLayout_2;
    QComboBox *comboBox_3;
    QPushButton *pushButton_3;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *pushButton_10;
    QHBoxLayout *horizontalLayout_3;
    QComboBox *comboBox_4;
    QPushButton *pushButton_4;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_2;
    QSpacerItem *horizontalSpacer;
    QDoubleSpinBox *doubleSpinBox;
    QPushButton *pushButton_5;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_4;
    QSpacerItem *horizontalSpacer_2;
    QComboBox *comboBox_6;
    QPushButton *pushButton_8;
    QHBoxLayout *horizontalLayout;
    QLabel *label_5;
    QSpacerItem *horizontalSpacer_6;
    QSpinBox *spinBox;
    QSpacerItem *horizontalSpacer_5;
    QHBoxLayout *horizontalLayout_7;
    QProgressBar *progressBar;
    QPushButton *pushButton_9;
    QPushButton *pushButton_12;
    QSpacerItem *verticalSpacer;
    QWidget *tab_2;

    void setupUi(QWidget *MachiningDockWidget)
    {
        if (MachiningDockWidget->objectName().isEmpty())
            MachiningDockWidget->setObjectName(QString::fromUtf8("MachiningDockWidget"));
        MachiningDockWidget->resize(293, 530);
        verticalLayout = new QVBoxLayout(MachiningDockWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabWidget = new QTabWidget(MachiningDockWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        comboBox_5 = new QComboBox(tab);
        comboBox_5->addItem(QString());
        comboBox_5->addItem(QString());
        comboBox_5->addItem(QString());
        comboBox_5->addItem(QString());
        comboBox_5->addItem(QString());
        comboBox_5->addItem(QString());
        comboBox_5->setObjectName(QString::fromUtf8("comboBox_5"));
        comboBox_5->setMinimumSize(QSize(0, 25));
        comboBox_5->setMaximumSize(QSize(16777215, 25));
        QFont font;
        font.setPointSize(9);
        comboBox_5->setFont(font);
        comboBox_5->setStyleSheet(QString::fromUtf8("padding-left:3px;"));

        horizontalLayout_5->addWidget(comboBox_5);

        pushButton_7 = new QPushButton(tab);
        pushButton_7->setObjectName(QString::fromUtf8("pushButton_7"));
        pushButton_7->setMinimumSize(QSize(25, 25));
        pushButton_7->setMaximumSize(QSize(25, 25));
        pushButton_7->setFont(font);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/amwind/figure/am_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_7->setIcon(icon);

        horizontalLayout_5->addWidget(pushButton_7);


        verticalLayout_2->addLayout(horizontalLayout_5);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        pushButton = new QPushButton(tab);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setMinimumSize(QSize(25, 25));
        pushButton->setMaximumSize(QSize(25, 25));
        pushButton->setIcon(icon);

        gridLayout->addWidget(pushButton, 0, 2, 1, 1);

        lineEdit = new QLineEdit(tab);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        lineEdit->setMinimumSize(QSize(0, 25));
        lineEdit->setMaximumSize(QSize(16777215, 25));
        lineEdit->setFont(font);
        lineEdit->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        lineEdit->setReadOnly(true);

        gridLayout->addWidget(lineEdit, 0, 1, 1, 1);

        comboBox = new QComboBox(tab);
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setMaximumSize(QSize(65, 16777215));
        comboBox->setFont(font);
        comboBox->setStyleSheet(QString::fromUtf8("padding-left:3px;"));

        gridLayout->addWidget(comboBox, 1, 0, 1, 1);

        comboBox_2 = new QComboBox(tab);
        comboBox_2->addItem(QString());
        comboBox_2->addItem(QString());
        comboBox_2->setObjectName(QString::fromUtf8("comboBox_2"));
        comboBox_2->setMaximumSize(QSize(65, 16777215));
        comboBox_2->setFont(font);
        comboBox_2->setStyleSheet(QString::fromUtf8("padding-left:3px;"));

        gridLayout->addWidget(comboBox_2, 0, 0, 1, 1);

        lineEdit_2 = new QLineEdit(tab);
        lineEdit_2->setObjectName(QString::fromUtf8("lineEdit_2"));
        lineEdit_2->setMinimumSize(QSize(0, 25));
        lineEdit_2->setMaximumSize(QSize(16777215, 25));
        lineEdit_2->setFont(font);
        lineEdit_2->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        lineEdit_2->setReadOnly(true);

        gridLayout->addWidget(lineEdit_2, 1, 1, 1, 1);

        pushButton_2 = new QPushButton(tab);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setMinimumSize(QSize(25, 25));
        pushButton_2->setMaximumSize(QSize(25, 25));
        pushButton_2->setIcon(icon);

        gridLayout->addWidget(pushButton_2, 1, 2, 1, 1);


        verticalLayout_2->addLayout(gridLayout);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_3 = new QLabel(tab);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setFont(font);

        horizontalLayout_9->addWidget(label_3);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_4);

        pushButton_11 = new QPushButton(tab);
        pushButton_11->setObjectName(QString::fromUtf8("pushButton_11"));
        pushButton_11->setMinimumSize(QSize(25, 25));
        pushButton_11->setMaximumSize(QSize(25, 25));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/fem_wind/figure/fem_wind/unchecked.png"), QSize(), QIcon::Normal, QIcon::Off);
        icon1.addFile(QString::fromUtf8(":/fem_wind/figure/fem_wind/checked.png"), QSize(), QIcon::Normal, QIcon::On);
        pushButton_11->setIcon(icon1);
        pushButton_11->setCheckable(true);

        horizontalLayout_9->addWidget(pushButton_11);


        verticalLayout_2->addLayout(horizontalLayout_9);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        comboBox_3 = new QComboBox(tab);
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->setObjectName(QString::fromUtf8("comboBox_3"));
        comboBox_3->setFont(font);
        comboBox_3->setStyleSheet(QString::fromUtf8("padding-left:3px;"));

        horizontalLayout_2->addWidget(comboBox_3);

        pushButton_3 = new QPushButton(tab);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        pushButton_3->setMinimumSize(QSize(25, 25));
        pushButton_3->setMaximumSize(QSize(25, 25));
        pushButton_3->setIcon(icon);

        horizontalLayout_2->addWidget(pushButton_3);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label = new QLabel(tab);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFont(font);

        horizontalLayout_8->addWidget(label);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_3);

        pushButton_10 = new QPushButton(tab);
        pushButton_10->setObjectName(QString::fromUtf8("pushButton_10"));
        pushButton_10->setMinimumSize(QSize(25, 25));
        pushButton_10->setMaximumSize(QSize(25, 25));
        pushButton_10->setIcon(icon1);
        pushButton_10->setCheckable(true);

        horizontalLayout_8->addWidget(pushButton_10);


        verticalLayout_2->addLayout(horizontalLayout_8);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        comboBox_4 = new QComboBox(tab);
        comboBox_4->addItem(QString());
        comboBox_4->addItem(QString());
        comboBox_4->setObjectName(QString::fromUtf8("comboBox_4"));
        comboBox_4->setMinimumSize(QSize(0, 25));
        comboBox_4->setMaximumSize(QSize(16777215, 25));
        comboBox_4->setFont(font);
        comboBox_4->setStyleSheet(QString::fromUtf8("padding-left:3px;"));

        horizontalLayout_3->addWidget(comboBox_4);

        pushButton_4 = new QPushButton(tab);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));
        pushButton_4->setMinimumSize(QSize(25, 25));
        pushButton_4->setMaximumSize(QSize(25, 25));
        pushButton_4->setIcon(icon);

        horizontalLayout_3->addWidget(pushButton_4);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_2 = new QLabel(tab);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);

        horizontalLayout_4->addWidget(label_2);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);

        doubleSpinBox = new QDoubleSpinBox(tab);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        doubleSpinBox->setMinimumSize(QSize(60, 25));
        doubleSpinBox->setMaximumSize(QSize(16777215, 25));
        doubleSpinBox->setFont(font);
        doubleSpinBox->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox->setReadOnly(true);
        doubleSpinBox->setValue(0.100000000000000);

        horizontalLayout_4->addWidget(doubleSpinBox);

        pushButton_5 = new QPushButton(tab);
        pushButton_5->setObjectName(QString::fromUtf8("pushButton_5"));
        pushButton_5->setMinimumSize(QSize(25, 25));
        pushButton_5->setMaximumSize(QSize(25, 25));
        pushButton_5->setFont(font);
        pushButton_5->setIcon(icon);

        horizontalLayout_4->addWidget(pushButton_5);


        verticalLayout_2->addLayout(horizontalLayout_4);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_4 = new QLabel(tab);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFont(font);

        horizontalLayout_6->addWidget(label_4);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_2);

        comboBox_6 = new QComboBox(tab);
        comboBox_6->addItem(QString());
        comboBox_6->addItem(QString());
        comboBox_6->addItem(QString());
        comboBox_6->setObjectName(QString::fromUtf8("comboBox_6"));
        comboBox_6->setMinimumSize(QSize(0, 25));
        comboBox_6->setMaximumSize(QSize(16777215, 25));
        comboBox_6->setFont(font);
        comboBox_6->setStyleSheet(QString::fromUtf8("padding-left:3px;"));

        horizontalLayout_6->addWidget(comboBox_6);

        pushButton_8 = new QPushButton(tab);
        pushButton_8->setObjectName(QString::fromUtf8("pushButton_8"));
        pushButton_8->setMinimumSize(QSize(25, 25));
        pushButton_8->setMaximumSize(QSize(25, 25));
        pushButton_8->setIcon(icon);

        horizontalLayout_6->addWidget(pushButton_8);


        verticalLayout_2->addLayout(horizontalLayout_6);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_5 = new QLabel(tab);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setFont(font);

        horizontalLayout->addWidget(label_5);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_6);

        spinBox = new QSpinBox(tab);
        spinBox->setObjectName(QString::fromUtf8("spinBox"));
        spinBox->setMinimumSize(QSize(65, 25));
        spinBox->setMaximumSize(QSize(16777215, 25));
        spinBox->setFont(font);
        spinBox->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        spinBox->setValue(20);

        horizontalLayout->addWidget(spinBox);

        horizontalSpacer_5 = new QSpacerItem(30, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_5);


        verticalLayout_2->addLayout(horizontalLayout);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        progressBar = new QProgressBar(tab);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setMinimumSize(QSize(0, 25));
        progressBar->setMaximumSize(QSize(16777215, 25));
        progressBar->setFont(font);
        progressBar->setValue(0);

        horizontalLayout_7->addWidget(progressBar);

        pushButton_9 = new QPushButton(tab);
        pushButton_9->setObjectName(QString::fromUtf8("pushButton_9"));
        pushButton_9->setMinimumSize(QSize(25, 25));
        pushButton_9->setMaximumSize(QSize(25, 25));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/amwind/figure/am_wind/animation.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_9->setIcon(icon2);

        horizontalLayout_7->addWidget(pushButton_9);

        pushButton_12 = new QPushButton(tab);
        pushButton_12->setObjectName(QString::fromUtf8("pushButton_12"));
        pushButton_12->setMinimumSize(QSize(25, 25));
        pushButton_12->setMaximumSize(QSize(25, 25));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/new/prefix1/figure/database_wind/edit.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_12->setIcon(icon3);

        horizontalLayout_7->addWidget(pushButton_12);


        verticalLayout_2->addLayout(horizontalLayout_7);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/figure/machining/1.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab, icon4, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/figure/machining/2.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab_2, icon5, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(MachiningDockWidget);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MachiningDockWidget);
    } // setupUi

    void retranslateUi(QWidget *MachiningDockWidget)
    {
        MachiningDockWidget->setWindowTitle(QApplication::translate("MachiningDockWidget", "Form", nullptr));
        comboBox_5->setItemText(0, QApplication::translate("MachiningDockWidget", "Cook Membrane", nullptr));
        comboBox_5->setItemText(1, QApplication::translate("MachiningDockWidget", "Internally Pressurized Cylinder", nullptr));
        comboBox_5->setItemText(2, QApplication::translate("MachiningDockWidget", "Internally Pressurized Sphere", nullptr));
        comboBox_5->setItemText(3, QApplication::translate("MachiningDockWidget", "Punctured Disc", nullptr));
        comboBox_5->setItemText(4, QApplication::translate("MachiningDockWidget", "Taylor Bar Impact", nullptr));
        comboBox_5->setItemText(5, QApplication::translate("MachiningDockWidget", "ThermoMechanical Traction", nullptr));

        pushButton_7->setText(QString());
        pushButton->setText(QString());
        lineEdit->setText(QApplication::translate("MachiningDockWidget", "10,1,1", nullptr));
        comboBox->setItemText(0, QApplication::translate("MachiningDockWidget", "Tool Size", nullptr));
        comboBox->setItemText(1, QApplication::translate("MachiningDockWidget", "Tool Pos", nullptr));
        comboBox->setItemText(2, QApplication::translate("MachiningDockWidget", "Tool Type", nullptr));

        comboBox_2->setItemText(0, QApplication::translate("MachiningDockWidget", "Part Size", nullptr));
        comboBox_2->setItemText(1, QApplication::translate("MachiningDockWidget", "Part Pos", nullptr));

        lineEdit_2->setText(QApplication::translate("MachiningDockWidget", "1,1,1", nullptr));
        pushButton_2->setText(QString());
        label_3->setText(QApplication::translate("MachiningDockWidget", "Consitutive Laws", nullptr));
        pushButton_11->setText(QString());
        comboBox_3->setItemText(0, QApplication::translate("MachiningDockWidget", "von Mises", nullptr));
        comboBox_3->setItemText(1, QApplication::translate("MachiningDockWidget", "Simo", nullptr));
        comboBox_3->setItemText(2, QApplication::translate("MachiningDockWidget", "Johnson Cook", nullptr));
        comboBox_3->setItemText(3, QApplication::translate("MachiningDockWidget", "Baker", nullptr));

        pushButton_3->setText(QString());
        label->setText(QApplication::translate("MachiningDockWidget", "Boundary Conditions", nullptr));
        pushButton_10->setText(QString());
        comboBox_4->setItemText(0, QApplication::translate("MachiningDockWidget", "Displacement", nullptr));
        comboBox_4->setItemText(1, QApplication::translate("MachiningDockWidget", "Stress", nullptr));

        pushButton_4->setText(QString());
        label_2->setText(QApplication::translate("MachiningDockWidget", "Mesh", nullptr));
        pushButton_5->setText(QString());
        label_4->setText(QApplication::translate("MachiningDockWidget", "Solver", nullptr));
        comboBox_6->setItemText(0, QApplication::translate("MachiningDockWidget", "Elasticity", nullptr));
        comboBox_6->setItemText(1, QApplication::translate("MachiningDockWidget", "ElastoPlasticity", nullptr));
        comboBox_6->setItemText(2, QApplication::translate("MachiningDockWidget", "ViscoElastoPlasticity", nullptr));

        pushButton_8->setText(QString());
        label_5->setText(QApplication::translate("MachiningDockWidget", "Time Step", nullptr));
        pushButton_9->setText(QString());
        pushButton_12->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab), QString());
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QString());
    } // retranslateUi

};

namespace Ui {
    class MachiningDockWidget: public Ui_MachiningDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MACHININGDOCKWIDGET_H
