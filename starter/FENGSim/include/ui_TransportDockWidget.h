/********************************************************************************
** Form generated from reading UI file 'TransportDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRANSPORTDOCKWIDGET_H
#define UI_TRANSPORTDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TransportDockWidget
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *verticalLayout_2;
    QTableWidget *tableWidget;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QPushButton *pushButton_3;
    QSpacerItem *verticalSpacer;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label;
    QPushButton *pushButton_10;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_2;
    QDoubleSpinBox *doubleSpinBox;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_5;
    QDoubleSpinBox *doubleSpinBox_2;
    QHBoxLayout *horizontalLayout;
    QLabel *label_3;
    QPushButton *pushButton_5;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_4;
    QPushButton *pushButton_7;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_6;
    QPushButton *pushButton_11;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_7;
    QDoubleSpinBox *doubleSpinBox_3;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_8;
    QPushButton *pushButton_8;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_9;
    QPushButton *pushButton_9;
    QHBoxLayout *horizontalLayout_3;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *pushButton_6;
    QComboBox *comboBox;
    QPushButton *pushButton_4;
    QSpacerItem *verticalSpacer_2;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_5;
    QTableWidget *tableWidget_2;
    QSpacerItem *verticalSpacer_3;

    void setupUi(QWidget *TransportDockWidget)
    {
        if (TransportDockWidget->objectName().isEmpty())
            TransportDockWidget->setObjectName(QString::fromUtf8("TransportDockWidget"));
        TransportDockWidget->resize(399, 766);
        verticalLayout = new QVBoxLayout(TransportDockWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabWidget = new QTabWidget(TransportDockWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_3 = new QVBoxLayout(tab);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        tableWidget = new QTableWidget(tab);
        if (tableWidget->columnCount() < 1)
            tableWidget->setColumnCount(1);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        tableWidget->setHorizontalHeaderItem(0, __qtablewidgetitem);
        if (tableWidget->rowCount() < 2)
            tableWidget->setRowCount(2);
        QFont font;
        font.setPointSize(9);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        __qtablewidgetitem1->setFont(font);
        tableWidget->setVerticalHeaderItem(0, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        __qtablewidgetitem2->setFont(font);
        tableWidget->setVerticalHeaderItem(1, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        __qtablewidgetitem3->setFont(font);
        tableWidget->setItem(0, 0, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        tableWidget->setItem(1, 0, __qtablewidgetitem4);
        tableWidget->setObjectName(QString::fromUtf8("tableWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(1);
        sizePolicy.setHeightForWidth(tableWidget->sizePolicy().hasHeightForWidth());
        tableWidget->setSizePolicy(sizePolicy);
        tableWidget->setMaximumSize(QSize(16777215, 60));
        tableWidget->setFont(font);
        tableWidget->setStyleSheet(QString::fromUtf8("QTableWidget::item {\n"
"    padding: 1px;\n"
"    border: 0px;\n"
"    color: white;\n"
"    background-color: black;\n"
"}\n"
"\n"
"QTableWidget::item:pressed, QListView::item:pressed, QTreeView::item:pressed  {\n"
"    background-color: black;\n"
"    color: white;\n"
"}\n"
"\n"
"QTableWidget::item:selected:active, QTreeView::item:selected:active, QListView::item:selected:active  {\n"
"    background-color: black;\n"
"    color: white;\n"
"}"));
        tableWidget->setLineWidth(0);
        tableWidget->horizontalHeader()->setVisible(false);
        tableWidget->horizontalHeader()->setCascadingSectionResizes(false);
        tableWidget->horizontalHeader()->setHighlightSections(false);
        tableWidget->horizontalHeader()->setProperty("showSortIndicator", QVariant(false));
        tableWidget->horizontalHeader()->setStretchLastSection(true);
        tableWidget->verticalHeader()->setVisible(true);
        tableWidget->verticalHeader()->setCascadingSectionResizes(false);
        tableWidget->verticalHeader()->setHighlightSections(false);
        tableWidget->verticalHeader()->setProperty("showSortIndicator", QVariant(false));
        tableWidget->verticalHeader()->setStretchLastSection(false);

        verticalLayout_2->addWidget(tableWidget);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        pushButton = new QPushButton(tab);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setMinimumSize(QSize(25, 25));
        pushButton->setMaximumSize(QSize(25, 25));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/amwind/figure/am_wind/open.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton->setIcon(icon);

        horizontalLayout_2->addWidget(pushButton);

        pushButton_2 = new QPushButton(tab);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setMinimumSize(QSize(25, 25));
        pushButton_2->setMaximumSize(QSize(25, 25));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/save.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_2->setIcon(icon1);

        horizontalLayout_2->addWidget(pushButton_2);

        pushButton_3 = new QPushButton(tab);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        pushButton_3->setMinimumSize(QSize(25, 25));
        pushButton_3->setMaximumSize(QSize(25, 25));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/amwind/figure/am_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_3->setIcon(icon2);

        horizontalLayout_2->addWidget(pushButton_3);


        verticalLayout_2->addLayout(horizontalLayout_2);


        verticalLayout_3->addLayout(verticalLayout_2);

        verticalSpacer = new QSpacerItem(20, 300, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/new/prefix1/cad.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab, icon3, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_4 = new QVBoxLayout(tab_2);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        label = new QLabel(tab_2);
        label->setObjectName(QString::fromUtf8("label"));
        QFont font1;
        font1.setPointSize(9);
        font1.setBold(false);
        font1.setWeight(50);
        label->setFont(font1);

        horizontalLayout_11->addWidget(label);

        pushButton_10 = new QPushButton(tab_2);
        pushButton_10->setObjectName(QString::fromUtf8("pushButton_10"));
        pushButton_10->setMinimumSize(QSize(25, 25));
        pushButton_10->setMaximumSize(QSize(25, 25));
        pushButton_10->setIcon(icon2);

        horizontalLayout_11->addWidget(pushButton_10);


        verticalLayout_4->addLayout(horizontalLayout_11);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_2 = new QLabel(tab_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);

        horizontalLayout_9->addWidget(label_2);

        doubleSpinBox = new QDoubleSpinBox(tab_2);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        doubleSpinBox->setMinimumSize(QSize(0, 25));
        doubleSpinBox->setMaximumSize(QSize(16777215, 25));
        doubleSpinBox->setFont(font);
        doubleSpinBox->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox->setValue(90.000000000000000);

        horizontalLayout_9->addWidget(doubleSpinBox);


        verticalLayout_4->addLayout(horizontalLayout_9);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_5 = new QLabel(tab_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setFont(font);

        horizontalLayout_10->addWidget(label_5);

        doubleSpinBox_2 = new QDoubleSpinBox(tab_2);
        doubleSpinBox_2->setObjectName(QString::fromUtf8("doubleSpinBox_2"));
        doubleSpinBox_2->setMinimumSize(QSize(0, 25));
        doubleSpinBox_2->setMaximumSize(QSize(16777215, 25));
        doubleSpinBox_2->setFont(font);
        doubleSpinBox_2->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox_2->setValue(19.800000000000001);

        horizontalLayout_10->addWidget(doubleSpinBox_2);


        verticalLayout_4->addLayout(horizontalLayout_10);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_3 = new QLabel(tab_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setFont(font);

        horizontalLayout->addWidget(label_3);

        pushButton_5 = new QPushButton(tab_2);
        pushButton_5->setObjectName(QString::fromUtf8("pushButton_5"));
        pushButton_5->setMinimumSize(QSize(25, 25));
        pushButton_5->setMaximumSize(QSize(25, 25));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/fem_wind/figure/fem_wind/unchecked.png"), QSize(), QIcon::Normal, QIcon::Off);
        icon4.addFile(QString::fromUtf8(":/fem_wind/figure/fem_wind/checked.png"), QSize(), QIcon::Normal, QIcon::On);
        pushButton_5->setIcon(icon4);
        pushButton_5->setCheckable(true);

        horizontalLayout->addWidget(pushButton_5);


        verticalLayout_4->addLayout(horizontalLayout);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_4 = new QLabel(tab_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFont(font);

        horizontalLayout_4->addWidget(label_4);

        pushButton_7 = new QPushButton(tab_2);
        pushButton_7->setObjectName(QString::fromUtf8("pushButton_7"));
        pushButton_7->setMinimumSize(QSize(25, 25));
        pushButton_7->setMaximumSize(QSize(25, 25));
        pushButton_7->setIcon(icon4);
        pushButton_7->setCheckable(true);

        horizontalLayout_4->addWidget(pushButton_7);


        verticalLayout_4->addLayout(horizontalLayout_4);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_6 = new QLabel(tab_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setFont(font1);

        horizontalLayout_7->addWidget(label_6);

        pushButton_11 = new QPushButton(tab_2);
        pushButton_11->setObjectName(QString::fromUtf8("pushButton_11"));
        pushButton_11->setMinimumSize(QSize(25, 25));
        pushButton_11->setMaximumSize(QSize(25, 25));
        pushButton_11->setIcon(icon2);

        horizontalLayout_7->addWidget(pushButton_11);


        verticalLayout_4->addLayout(horizontalLayout_7);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label_7 = new QLabel(tab_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setFont(font);

        horizontalLayout_12->addWidget(label_7);

        doubleSpinBox_3 = new QDoubleSpinBox(tab_2);
        doubleSpinBox_3->setObjectName(QString::fromUtf8("doubleSpinBox_3"));
        doubleSpinBox_3->setMinimumSize(QSize(0, 25));
        doubleSpinBox_3->setMaximumSize(QSize(16777215, 25));
        doubleSpinBox_3->setFont(font);
        doubleSpinBox_3->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox_3->setDecimals(4);
        doubleSpinBox_3->setValue(0.870000000000000);

        horizontalLayout_12->addWidget(doubleSpinBox_3);


        verticalLayout_4->addLayout(horizontalLayout_12);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_8 = new QLabel(tab_2);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setFont(font);

        horizontalLayout_5->addWidget(label_8);

        pushButton_8 = new QPushButton(tab_2);
        pushButton_8->setObjectName(QString::fromUtf8("pushButton_8"));
        pushButton_8->setMinimumSize(QSize(25, 25));
        pushButton_8->setMaximumSize(QSize(25, 25));
        pushButton_8->setIcon(icon4);
        pushButton_8->setCheckable(true);

        horizontalLayout_5->addWidget(pushButton_8);


        verticalLayout_4->addLayout(horizontalLayout_5);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_9 = new QLabel(tab_2);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setFont(font);

        horizontalLayout_6->addWidget(label_9);

        pushButton_9 = new QPushButton(tab_2);
        pushButton_9->setObjectName(QString::fromUtf8("pushButton_9"));
        pushButton_9->setMinimumSize(QSize(25, 25));
        pushButton_9->setMaximumSize(QSize(25, 25));
        pushButton_9->setIcon(icon4);
        pushButton_9->setCheckable(true);

        horizontalLayout_6->addWidget(pushButton_9);


        verticalLayout_4->addLayout(horizontalLayout_6);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_3);

        pushButton_6 = new QPushButton(tab_2);
        pushButton_6->setObjectName(QString::fromUtf8("pushButton_6"));
        pushButton_6->setMinimumSize(QSize(25, 25));
        pushButton_6->setMaximumSize(QSize(25, 25));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/selection_domain.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_6->setIcon(icon5);
        pushButton_6->setCheckable(true);

        horizontalLayout_3->addWidget(pushButton_6);

        comboBox = new QComboBox(tab_2);
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setFont(font);

        horizontalLayout_3->addWidget(comboBox);

        pushButton_4 = new QPushButton(tab_2);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));
        pushButton_4->setMinimumSize(QSize(25, 25));
        pushButton_4->setMaximumSize(QSize(25, 25));
        pushButton_4->setIcon(icon2);

        horizontalLayout_3->addWidget(pushButton_4);


        verticalLayout_4->addLayout(horizontalLayout_3);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_2);

        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/new/prefix1/source.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab_2, icon6, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        verticalLayout_5 = new QVBoxLayout(tab_3);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        tableWidget_2 = new QTableWidget(tab_3);
        if (tableWidget_2->columnCount() < 1)
            tableWidget_2->setColumnCount(1);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        tableWidget_2->setHorizontalHeaderItem(0, __qtablewidgetitem5);
        if (tableWidget_2->rowCount() < 9)
            tableWidget_2->setRowCount(9);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        __qtablewidgetitem6->setFont(font);
        tableWidget_2->setVerticalHeaderItem(0, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        __qtablewidgetitem7->setFont(font);
        tableWidget_2->setVerticalHeaderItem(1, __qtablewidgetitem7);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        __qtablewidgetitem8->setFont(font);
        tableWidget_2->setVerticalHeaderItem(2, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        __qtablewidgetitem9->setFont(font);
        tableWidget_2->setVerticalHeaderItem(3, __qtablewidgetitem9);
        QTableWidgetItem *__qtablewidgetitem10 = new QTableWidgetItem();
        __qtablewidgetitem10->setFont(font);
        tableWidget_2->setVerticalHeaderItem(4, __qtablewidgetitem10);
        QTableWidgetItem *__qtablewidgetitem11 = new QTableWidgetItem();
        __qtablewidgetitem11->setFont(font);
        tableWidget_2->setVerticalHeaderItem(5, __qtablewidgetitem11);
        QTableWidgetItem *__qtablewidgetitem12 = new QTableWidgetItem();
        __qtablewidgetitem12->setFont(font);
        tableWidget_2->setVerticalHeaderItem(6, __qtablewidgetitem12);
        QTableWidgetItem *__qtablewidgetitem13 = new QTableWidgetItem();
        __qtablewidgetitem13->setFont(font);
        tableWidget_2->setVerticalHeaderItem(7, __qtablewidgetitem13);
        QTableWidgetItem *__qtablewidgetitem14 = new QTableWidgetItem();
        __qtablewidgetitem14->setFont(font);
        tableWidget_2->setVerticalHeaderItem(8, __qtablewidgetitem14);
        QTableWidgetItem *__qtablewidgetitem15 = new QTableWidgetItem();
        __qtablewidgetitem15->setFont(font);
        tableWidget_2->setItem(0, 0, __qtablewidgetitem15);
        QTableWidgetItem *__qtablewidgetitem16 = new QTableWidgetItem();
        __qtablewidgetitem16->setFont(font);
        tableWidget_2->setItem(1, 0, __qtablewidgetitem16);
        QTableWidgetItem *__qtablewidgetitem17 = new QTableWidgetItem();
        __qtablewidgetitem17->setFont(font);
        tableWidget_2->setItem(2, 0, __qtablewidgetitem17);
        QTableWidgetItem *__qtablewidgetitem18 = new QTableWidgetItem();
        __qtablewidgetitem18->setFont(font);
        tableWidget_2->setItem(3, 0, __qtablewidgetitem18);
        QTableWidgetItem *__qtablewidgetitem19 = new QTableWidgetItem();
        __qtablewidgetitem19->setFont(font);
        tableWidget_2->setItem(4, 0, __qtablewidgetitem19);
        QTableWidgetItem *__qtablewidgetitem20 = new QTableWidgetItem();
        __qtablewidgetitem20->setFont(font);
        tableWidget_2->setItem(5, 0, __qtablewidgetitem20);
        QTableWidgetItem *__qtablewidgetitem21 = new QTableWidgetItem();
        __qtablewidgetitem21->setFont(font);
        tableWidget_2->setItem(6, 0, __qtablewidgetitem21);
        QTableWidgetItem *__qtablewidgetitem22 = new QTableWidgetItem();
        __qtablewidgetitem22->setFont(font);
        tableWidget_2->setItem(7, 0, __qtablewidgetitem22);
        QTableWidgetItem *__qtablewidgetitem23 = new QTableWidgetItem();
        __qtablewidgetitem23->setFont(font);
        tableWidget_2->setItem(8, 0, __qtablewidgetitem23);
        tableWidget_2->setObjectName(QString::fromUtf8("tableWidget_2"));
        tableWidget_2->setMinimumSize(QSize(0, 270));
        tableWidget_2->setMaximumSize(QSize(16777215, 270));
        tableWidget_2->setFont(font);
        tableWidget_2->setStyleSheet(QString::fromUtf8("QTableWidget::item {\n"
"    padding: 1px;\n"
"    border: 0px;\n"
"    color: white;\n"
"    background-color: black;\n"
"}\n"
"\n"
"QTableWidget::item:pressed, QListView::item:pressed, QTreeView::item:pressed  {\n"
"    background-color: black;\n"
"    color: white;\n"
"}\n"
"\n"
"QTableWidget::item:selected:active, QTreeView::item:selected:active, QListView::item:selected:active  {\n"
"    background-color: black;\n"
"    color: white;\n"
"}"));
        tableWidget_2->horizontalHeader()->setVisible(false);
        tableWidget_2->horizontalHeader()->setStretchLastSection(true);
        tableWidget_2->verticalHeader()->setStretchLastSection(true);

        verticalLayout_5->addWidget(tableWidget_2);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_3);

        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/new/prefix1/data.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab_3, icon7, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(TransportDockWidget);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(TransportDockWidget);
    } // setupUi

    void retranslateUi(QWidget *TransportDockWidget)
    {
        TransportDockWidget->setWindowTitle(QApplication::translate("TransportDockWidget", "Form", nullptr));
        QTableWidgetItem *___qtablewidgetitem = tableWidget->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("TransportDockWidget", "n", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = tableWidget->verticalHeaderItem(0);
        ___qtablewidgetitem1->setText(QApplication::translate("TransportDockWidget", "level", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = tableWidget->verticalHeaderItem(1);
        ___qtablewidgetitem2->setText(QApplication::translate("TransportDockWidget", "number", nullptr));

        const bool __sortingEnabled = tableWidget->isSortingEnabled();
        tableWidget->setSortingEnabled(false);
        QTableWidgetItem *___qtablewidgetitem3 = tableWidget->item(0, 0);
        ___qtablewidgetitem3->setText(QApplication::translate("TransportDockWidget", "3", nullptr));
        QTableWidgetItem *___qtablewidgetitem4 = tableWidget->item(1, 0);
        ___qtablewidgetitem4->setText(QApplication::translate("TransportDockWidget", "8", nullptr));
        tableWidget->setSortingEnabled(__sortingEnabled);

        pushButton->setText(QString());
        pushButton_2->setText(QString());
        pushButton_3->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab), QString());
        label->setText(QApplication::translate("TransportDockWidget", "1. Sphere", nullptr));
        pushButton_10->setText(QString());
        label_2->setText(QApplication::translate("TransportDockWidget", "Pu %", nullptr));
        label_5->setText(QApplication::translate("TransportDockWidget", "density", nullptr));
        label_3->setText(QApplication::translate("TransportDockWidget", "Pu 239", nullptr));
        pushButton_5->setText(QString());
        label_4->setText(QApplication::translate("TransportDockWidget", "Pu 240", nullptr));
        pushButton_7->setText(QString());
        label_6->setText(QApplication::translate("TransportDockWidget", "2. Detector", nullptr));
        pushButton_11->setText(QString());
        label_7->setText(QApplication::translate("TransportDockWidget", "density", nullptr));
        label_8->setText(QApplication::translate("TransportDockWidget", "H element", nullptr));
        pushButton_8->setText(QString());
        label_9->setText(QApplication::translate("TransportDockWidget", "C element", nullptr));
        pushButton_9->setText(QString());
        pushButton_6->setText(QString());
        comboBox->setItemText(0, QApplication::translate("TransportDockWidget", "Geant4", nullptr));
        comboBox->setItemText(1, QApplication::translate("TransportDockWidget", "OpenMC", nullptr));
        comboBox->setItemText(2, QApplication::translate("TransportDockWidget", "MCNP", nullptr));

        pushButton_4->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QString());
        QTableWidgetItem *___qtablewidgetitem5 = tableWidget_2->horizontalHeaderItem(0);
        ___qtablewidgetitem5->setText(QApplication::translate("TransportDockWidget", "value", nullptr));
        QTableWidgetItem *___qtablewidgetitem6 = tableWidget_2->verticalHeaderItem(0);
        ___qtablewidgetitem6->setText(QApplication::translate("TransportDockWidget", "s_F", nullptr));
        QTableWidgetItem *___qtablewidgetitem7 = tableWidget_2->verticalHeaderItem(1);
        ___qtablewidgetitem7->setText(QApplication::translate("TransportDockWidget", "s_M", nullptr));
        QTableWidgetItem *___qtablewidgetitem8 = tableWidget_2->verticalHeaderItem(2);
        ___qtablewidgetitem8->setText(QApplication::translate("TransportDockWidget", "s_A", nullptr));
        QTableWidgetItem *___qtablewidgetitem9 = tableWidget_2->verticalHeaderItem(3);
        ___qtablewidgetitem9->setText(QApplication::translate("TransportDockWidget", "epsi", nullptr));
        QTableWidgetItem *___qtablewidgetitem10 = tableWidget_2->verticalHeaderItem(4);
        ___qtablewidgetitem10->setText(QApplication::translate("TransportDockWidget", "C_S", nullptr));
        QTableWidgetItem *___qtablewidgetitem11 = tableWidget_2->verticalHeaderItem(5);
        ___qtablewidgetitem11->setText(QApplication::translate("TransportDockWidget", "C_D", nullptr));
        QTableWidgetItem *___qtablewidgetitem12 = tableWidget_2->verticalHeaderItem(6);
        ___qtablewidgetitem12->setText(QApplication::translate("TransportDockWidget", "C_T", nullptr));
        QTableWidgetItem *___qtablewidgetitem13 = tableWidget_2->verticalHeaderItem(7);
        ___qtablewidgetitem13->setText(QApplication::translate("TransportDockWidget", "C_Q", nullptr));
        QTableWidgetItem *___qtablewidgetitem14 = tableWidget_2->verticalHeaderItem(8);
        ___qtablewidgetitem14->setText(QApplication::translate("TransportDockWidget", "m", nullptr));

        const bool __sortingEnabled1 = tableWidget_2->isSortingEnabled();
        tableWidget_2->setSortingEnabled(false);
        QTableWidgetItem *___qtablewidgetitem15 = tableWidget_2->item(0, 0);
        ___qtablewidgetitem15->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem16 = tableWidget_2->item(1, 0);
        ___qtablewidgetitem16->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem17 = tableWidget_2->item(2, 0);
        ___qtablewidgetitem17->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem18 = tableWidget_2->item(3, 0);
        ___qtablewidgetitem18->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem19 = tableWidget_2->item(4, 0);
        ___qtablewidgetitem19->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem20 = tableWidget_2->item(5, 0);
        ___qtablewidgetitem20->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem21 = tableWidget_2->item(6, 0);
        ___qtablewidgetitem21->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem22 = tableWidget_2->item(7, 0);
        ___qtablewidgetitem22->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        QTableWidgetItem *___qtablewidgetitem23 = tableWidget_2->item(8, 0);
        ___qtablewidgetitem23->setText(QApplication::translate("TransportDockWidget", "0", nullptr));
        tableWidget_2->setSortingEnabled(__sortingEnabled1);

        tabWidget->setTabText(tabWidget->indexOf(tab_3), QString());
    } // retranslateUi

};

namespace Ui {
    class TransportDockWidget: public Ui_TransportDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRANSPORTDOCKWIDGET_H
