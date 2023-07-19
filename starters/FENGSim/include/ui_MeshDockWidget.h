/********************************************************************************
** Form generated from reading UI file 'MeshDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MESHDOCKWIDGET_H
#define UI_MESHDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MeshDockWidget
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_3;
    QGridLayout *gridLayout_2;
    QDoubleSpinBox *doubleSpinBox;
    QLabel *label;
    QPushButton *pushButton;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *MeshDockWidget)
    {
        if (MeshDockWidget->objectName().isEmpty())
            MeshDockWidget->setObjectName(QString::fromUtf8("MeshDockWidget"));
        MeshDockWidget->resize(205, 353);
        verticalLayout = new QVBoxLayout(MeshDockWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabWidget = new QTabWidget(MeshDockWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_3 = new QVBoxLayout(tab);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        doubleSpinBox = new QDoubleSpinBox(tab);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(doubleSpinBox->sizePolicy().hasHeightForWidth());
        doubleSpinBox->setSizePolicy(sizePolicy);
        doubleSpinBox->setMinimumSize(QSize(80, 25));
        doubleSpinBox->setMaximumSize(QSize(80, 25));
        QFont font;
        font.setPointSize(9);
        doubleSpinBox->setFont(font);
        doubleSpinBox->setStyleSheet(QString::fromUtf8("padding-left: 3px;"));
        doubleSpinBox->setMaximum(10000.000000000000000);

        gridLayout_2->addWidget(doubleSpinBox, 0, 1, 1, 1);

        label = new QLabel(tab);
        label->setObjectName(QString::fromUtf8("label"));
        label->setMinimumSize(QSize(0, 25));
        label->setMaximumSize(QSize(16777215, 25));
        label->setFont(font);

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

        pushButton = new QPushButton(tab);
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

        gridLayout_2->addWidget(pushButton, 0, 2, 1, 1);


        verticalLayout_3->addLayout(gridLayout_2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/mesh.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab, icon1, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(MeshDockWidget);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MeshDockWidget);
    } // setupUi

    void retranslateUi(QWidget *MeshDockWidget)
    {
        MeshDockWidget->setWindowTitle(QApplication::translate("MeshDockWidget", "Form", nullptr));
        label->setText(QApplication::translate("MeshDockWidget", "Size:", nullptr));
        pushButton->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab), QString());
    } // retranslateUi

};

namespace Ui {
    class MeshDockWidget: public Ui_MeshDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MESHDOCKWIDGET_H
