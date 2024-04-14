/********************************************************************************
** Form generated from reading UI file 'CADDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CADDOCKWIDGET_H
#define UI_CADDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CADDockWidget
{
public:
    QAction *actionSphere;
    QAction *actionCube;
    QAction *actionTorus;
    QAction *actionCone;
    QAction *actionCylinder;
    QAction *actionPoint;
    QAction *actionLine;
    QAction *actionSquare;
    QAction *actionUnion;
    QAction *actionSection;
    QAction *actionCut;
    QAction *actionSweep;
    QAction *actionExtrude;
    QAction *actionMirror;
    QAction *actionSelectFace;
    QAction *actionSelectDomain;
    QAction *actionPart1;
    QAction *actionPart2;
    QVBoxLayout *verticalLayout_5;
    QGridLayout *gridLayout;
    QPushButton *pushButton_4;
    QSpacerItem *horizontalSpacer_2;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton_9;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QPushButton *pushButton_3;
    QPushButton *pushButton_7;
    QPushButton *pushButton_6;
    QPushButton *pushButton_5;
    QPushButton *pushButton_13;
    QPushButton *pushButton_0;
    QSpacerItem *verticalSpacer_4;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_2;
    QGridLayout *gridLayout_3;
    QPushButton *pushButton_33;
    QComboBox *comboBox_2;
    QLabel *label_4;
    QGridLayout *gridLayout_4;
    QLabel *label;
    QDoubleSpinBox *doubleSpinBox_4;
    QLabel *label_3;
    QDoubleSpinBox *doubleSpinBox_5;
    QDoubleSpinBox *doubleSpinBox_6;
    QLabel *label_2;
    QVBoxLayout *verticalLayout;
    QSpacerItem *verticalSpacer;
    QSpacerItem *verticalSpacer_3;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_3;
    QGridLayout *gridLayout_5;
    QVBoxLayout *verticalLayout_4;
    QSpacerItem *verticalSpacer_6;
    QGridLayout *gridLayout_11;
    QLabel *label_7;
    QLabel *label_8;
    QDoubleSpinBox *doubleSpinBox_2;
    QDoubleSpinBox *doubleSpinBox_3;
    QDoubleSpinBox *doubleSpinBox;
    QLabel *label_6;
    QPushButton *pushButton_36;
    QLabel *label_5;
    QComboBox *comboBox;
    QSpacerItem *verticalSpacer_5;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_6;
    QHBoxLayout *horizontalLayout;
    QComboBox *comboBox_4;
    QPushButton *pushButton_10;
    QPushButton *pushButton_11;
    QGridLayout *gridLayout_2;
    QDoubleSpinBox *doubleSpinBox_8;
    QDoubleSpinBox *doubleSpinBox_7;
    QPushButton *pushButton_8;
    QComboBox *comboBox_3;
    QLabel *label_9;
    QLabel *label_10;
    QDoubleSpinBox *doubleSpinBox_9;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QWidget *CADDockWidget)
    {
        if (CADDockWidget->objectName().isEmpty())
            CADDockWidget->setObjectName(QString::fromUtf8("CADDockWidget"));
        CADDockWidget->resize(205, 379);
        CADDockWidget->setWindowOpacity(1.000000000000000);
        CADDockWidget->setStyleSheet(QString::fromUtf8("border: 0px;"));
        actionSphere = new QAction(CADDockWidget);
        actionSphere->setObjectName(QString::fromUtf8("actionSphere"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/ball.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSphere->setIcon(icon);
        QFont font;
        font.setPointSize(10);
        actionSphere->setFont(font);
        actionCube = new QAction(CADDockWidget);
        actionCube->setObjectName(QString::fromUtf8("actionCube"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/box.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCube->setIcon(icon1);
        actionCube->setFont(font);
        actionTorus = new QAction(CADDockWidget);
        actionTorus->setObjectName(QString::fromUtf8("actionTorus"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/torus.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTorus->setIcon(icon2);
        actionTorus->setFont(font);
        actionCone = new QAction(CADDockWidget);
        actionCone->setObjectName(QString::fromUtf8("actionCone"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/cone.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCone->setIcon(icon3);
        actionCone->setFont(font);
        actionCylinder = new QAction(CADDockWidget);
        actionCylinder->setObjectName(QString::fromUtf8("actionCylinder"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/cylinder.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCylinder->setIcon(icon4);
        actionCylinder->setFont(font);
        actionPoint = new QAction(CADDockWidget);
        actionPoint->setObjectName(QString::fromUtf8("actionPoint"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/point.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPoint->setIcon(icon5);
        actionPoint->setFont(font);
        actionLine = new QAction(CADDockWidget);
        actionLine->setObjectName(QString::fromUtf8("actionLine"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/line.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionLine->setIcon(icon6);
        actionLine->setFont(font);
        actionSquare = new QAction(CADDockWidget);
        actionSquare->setObjectName(QString::fromUtf8("actionSquare"));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/plane.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSquare->setIcon(icon7);
        actionSquare->setFont(font);
        actionUnion = new QAction(CADDockWidget);
        actionUnion->setObjectName(QString::fromUtf8("actionUnion"));
        actionUnion->setCheckable(false);
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/union.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionUnion->setIcon(icon8);
        actionUnion->setFont(font);
        actionSection = new QAction(CADDockWidget);
        actionSection->setObjectName(QString::fromUtf8("actionSection"));
        actionSection->setCheckable(false);
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/section.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSection->setIcon(icon9);
        actionSection->setFont(font);
        actionCut = new QAction(CADDockWidget);
        actionCut->setObjectName(QString::fromUtf8("actionCut"));
        actionCut->setCheckable(false);
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/cut.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCut->setIcon(icon10);
        actionCut->setFont(font);
        actionSweep = new QAction(CADDockWidget);
        actionSweep->setObjectName(QString::fromUtf8("actionSweep"));
        actionSweep->setCheckable(true);
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/sweep.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSweep->setIcon(icon11);
        actionSweep->setFont(font);
        actionExtrude = new QAction(CADDockWidget);
        actionExtrude->setObjectName(QString::fromUtf8("actionExtrude"));
        actionExtrude->setCheckable(true);
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/extrude.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionExtrude->setIcon(icon12);
        actionExtrude->setFont(font);
        actionMirror = new QAction(CADDockWidget);
        actionMirror->setObjectName(QString::fromUtf8("actionMirror"));
        actionMirror->setCheckable(true);
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/mirror.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionMirror->setIcon(icon13);
        actionMirror->setFont(font);
        actionSelectFace = new QAction(CADDockWidget);
        actionSelectFace->setObjectName(QString::fromUtf8("actionSelectFace"));
        actionSelectFace->setCheckable(true);
        QIcon icon14;
        icon14.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/selection_face.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSelectFace->setIcon(icon14);
        actionSelectFace->setFont(font);
        actionSelectDomain = new QAction(CADDockWidget);
        actionSelectDomain->setObjectName(QString::fromUtf8("actionSelectDomain"));
        actionSelectDomain->setCheckable(true);
        QIcon icon15;
        icon15.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/selection_domain.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSelectDomain->setIcon(icon15);
        actionSelectDomain->setFont(font);
        actionPart1 = new QAction(CADDockWidget);
        actionPart1->setObjectName(QString::fromUtf8("actionPart1"));
        actionPart1->setCheckable(true);
        QIcon icon16;
        icon16.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/part1.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPart1->setIcon(icon16);
        actionPart1->setFont(font);
        actionPart2 = new QAction(CADDockWidget);
        actionPart2->setObjectName(QString::fromUtf8("actionPart2"));
        actionPart2->setCheckable(true);
        QIcon icon17;
        icon17.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/part2.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPart2->setIcon(icon17);
        actionPart2->setFont(font);
        verticalLayout_5 = new QVBoxLayout(CADDockWidget);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(9, 9, 9, 9);
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setHorizontalSpacing(2);
        gridLayout->setVerticalSpacing(3);
        pushButton_4 = new QPushButton(CADDockWidget);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));
        pushButton_4->setMinimumSize(QSize(32, 32));
        pushButton_4->setMaximumSize(QSize(32, 32));
        pushButton_4->setFont(font);
        pushButton_4->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}"));
        pushButton_4->setIcon(icon);
        pushButton_4->setIconSize(QSize(20, 20));
        pushButton_4->setFlat(true);

        gridLayout->addWidget(pushButton_4, 0, 4, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 0, 0, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 8, 1, 1);

        pushButton_9 = new QPushButton(CADDockWidget);
        pushButton_9->setObjectName(QString::fromUtf8("pushButton_9"));
        pushButton_9->setMinimumSize(QSize(32, 32));
        pushButton_9->setMaximumSize(QSize(32, 32));
        pushButton_9->setFont(font);
        pushButton_9->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}"));
        QIcon icon18;
        icon18.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/delete.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_9->setIcon(icon18);
        pushButton_9->setIconSize(QSize(20, 20));
        pushButton_9->setFlat(true);

        gridLayout->addWidget(pushButton_9, 0, 5, 1, 1);

        pushButton = new QPushButton(CADDockWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(pushButton->sizePolicy().hasHeightForWidth());
        pushButton->setSizePolicy(sizePolicy);
        pushButton->setMinimumSize(QSize(32, 32));
        pushButton->setMaximumSize(QSize(32, 32));
        pushButton->setFont(font);
        pushButton->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}\n"
""));
        pushButton->setIcon(icon5);
        pushButton->setIconSize(QSize(20, 20));
        pushButton->setFlat(true);

        gridLayout->addWidget(pushButton, 0, 1, 1, 1);

        pushButton_2 = new QPushButton(CADDockWidget);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setMinimumSize(QSize(32, 32));
        pushButton_2->setMaximumSize(QSize(32, 32));
        pushButton_2->setFont(font);
        pushButton_2->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}\n"
""));
        QIcon icon19;
        icon19.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/curve.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_2->setIcon(icon19);
        pushButton_2->setIconSize(QSize(20, 20));
        pushButton_2->setFlat(true);

        gridLayout->addWidget(pushButton_2, 0, 2, 1, 1);

        pushButton_3 = new QPushButton(CADDockWidget);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        pushButton_3->setMinimumSize(QSize(32, 32));
        pushButton_3->setMaximumSize(QSize(32, 32));
        pushButton_3->setFont(font);
        pushButton_3->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}"));
        QIcon icon20;
        icon20.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/surface.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_3->setIcon(icon20);
        pushButton_3->setIconSize(QSize(20, 20));
        pushButton_3->setFlat(true);

        gridLayout->addWidget(pushButton_3, 0, 3, 1, 1);

        pushButton_7 = new QPushButton(CADDockWidget);
        pushButton_7->setObjectName(QString::fromUtf8("pushButton_7"));
        pushButton_7->setMinimumSize(QSize(32, 32));
        pushButton_7->setMaximumSize(QSize(32, 32));
        QIcon icon21;
        icon21.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/more.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_7->setIcon(icon21);
        pushButton_7->setIconSize(QSize(20, 20));

        gridLayout->addWidget(pushButton_7, 1, 5, 1, 1);

        pushButton_6 = new QPushButton(CADDockWidget);
        pushButton_6->setObjectName(QString::fromUtf8("pushButton_6"));
        pushButton_6->setMinimumSize(QSize(32, 32));
        pushButton_6->setMaximumSize(QSize(32, 32));
        pushButton_6->setIcon(icon21);
        pushButton_6->setIconSize(QSize(20, 20));

        gridLayout->addWidget(pushButton_6, 1, 4, 1, 1);

        pushButton_5 = new QPushButton(CADDockWidget);
        pushButton_5->setObjectName(QString::fromUtf8("pushButton_5"));
        pushButton_5->setMinimumSize(QSize(32, 32));
        pushButton_5->setMaximumSize(QSize(32, 32));
        pushButton_5->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}"));
        pushButton_5->setIcon(icon15);
        pushButton_5->setIconSize(QSize(20, 20));
        pushButton_5->setFlat(true);

        gridLayout->addWidget(pushButton_5, 1, 3, 1, 1);

        pushButton_13 = new QPushButton(CADDockWidget);
        pushButton_13->setObjectName(QString::fromUtf8("pushButton_13"));
        pushButton_13->setMinimumSize(QSize(32, 32));
        pushButton_13->setMaximumSize(QSize(32, 32));
        pushButton_13->setFont(font);
        pushButton_13->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}"));
        pushButton_13->setIcon(icon11);
        pushButton_13->setIconSize(QSize(20, 20));
        pushButton_13->setCheckable(true);
        pushButton_13->setFlat(true);

        gridLayout->addWidget(pushButton_13, 1, 2, 1, 1);

        pushButton_0 = new QPushButton(CADDockWidget);
        pushButton_0->setObjectName(QString::fromUtf8("pushButton_0"));
        pushButton_0->setMinimumSize(QSize(32, 32));
        pushButton_0->setMaximumSize(QSize(32, 32));
        pushButton_0->setFont(font);
        pushButton_0->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"}"));
        pushButton_0->setIcon(icon8);
        pushButton_0->setIconSize(QSize(20, 20));
        pushButton_0->setCheckable(true);
        pushButton_0->setFlat(true);

        gridLayout->addWidget(pushButton_0, 1, 1, 1, 1);


        verticalLayout_5->addLayout(gridLayout);

        verticalSpacer_4 = new QSpacerItem(20, 10, QSizePolicy::Minimum, QSizePolicy::Fixed);

        verticalLayout_5->addItem(verticalSpacer_4);

        tabWidget = new QTabWidget(CADDockWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setMinimumSize(QSize(0, 270));
        tabWidget->setFont(font);
        tabWidget->setAutoFillBackground(false);
        tabWidget->setStyleSheet(QString::fromUtf8(""));
        tabWidget->setInputMethodHints(Qt::ImhHiddenText);
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setIconSize(QSize(16, 16));
        tabWidget->setElideMode(Qt::ElideNone);
        tabWidget->setTabsClosable(false);
        tabWidget->setTabBarAutoHide(false);
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        pushButton_33 = new QPushButton(tab);
        pushButton_33->setObjectName(QString::fromUtf8("pushButton_33"));
        pushButton_33->setMinimumSize(QSize(25, 25));
        pushButton_33->setMaximumSize(QSize(25, 25));
        QFont font1;
        font1.setPointSize(8);
        pushButton_33->setFont(font1);
        pushButton_33->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"    padding: 2px;\n"
"}"));
        QIcon icon22;
        icon22.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_33->setIcon(icon22);
        pushButton_33->setAutoDefault(true);
        pushButton_33->setFlat(false);

        gridLayout_3->addWidget(pushButton_33, 1, 1, 1, 1);

        comboBox_2 = new QComboBox(tab);
        comboBox_2->setObjectName(QString::fromUtf8("comboBox_2"));
        comboBox_2->setMinimumSize(QSize(110, 25));
        comboBox_2->setMaximumSize(QSize(110, 25));
        QFont font2;
        font2.setPointSize(9);
        comboBox_2->setFont(font2);
        comboBox_2->setStyleSheet(QString::fromUtf8("padding-left:5px;"));

        gridLayout_3->addWidget(comboBox_2, 1, 0, 1, 1);

        label_4 = new QLabel(tab);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setMinimumSize(QSize(130, 25));
        label_4->setMaximumSize(QSize(120, 25));
        label_4->setFont(font2);

        gridLayout_3->addWidget(label_4, 0, 0, 1, 1);

        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setSizeConstraint(QLayout::SetDefaultConstraint);
        label = new QLabel(tab);
        label->setObjectName(QString::fromUtf8("label"));
        label->setMinimumSize(QSize(20, 25));
        label->setMaximumSize(QSize(20, 25));
        label->setFont(font2);

        gridLayout_4->addWidget(label, 0, 0, 1, 1);

        doubleSpinBox_4 = new QDoubleSpinBox(tab);
        doubleSpinBox_4->setObjectName(QString::fromUtf8("doubleSpinBox_4"));
        doubleSpinBox_4->setMinimumSize(QSize(80, 25));
        doubleSpinBox_4->setMaximumSize(QSize(80, 25));
        doubleSpinBox_4->setFont(font2);
        doubleSpinBox_4->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_4->setDecimals(5);
        doubleSpinBox_4->setMinimum(-99.989999999999995);

        gridLayout_4->addWidget(doubleSpinBox_4, 0, 1, 1, 1);

        label_3 = new QLabel(tab);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setMaximumSize(QSize(20, 25));
        label_3->setFont(font2);

        gridLayout_4->addWidget(label_3, 2, 0, 1, 1);

        doubleSpinBox_5 = new QDoubleSpinBox(tab);
        doubleSpinBox_5->setObjectName(QString::fromUtf8("doubleSpinBox_5"));
        doubleSpinBox_5->setMinimumSize(QSize(80, 25));
        doubleSpinBox_5->setMaximumSize(QSize(80, 25));
        doubleSpinBox_5->setFont(font2);
        doubleSpinBox_5->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_5->setDecimals(5);
        doubleSpinBox_5->setMinimum(-99.989999999999995);

        gridLayout_4->addWidget(doubleSpinBox_5, 1, 1, 1, 1);

        doubleSpinBox_6 = new QDoubleSpinBox(tab);
        doubleSpinBox_6->setObjectName(QString::fromUtf8("doubleSpinBox_6"));
        doubleSpinBox_6->setMinimumSize(QSize(80, 25));
        doubleSpinBox_6->setMaximumSize(QSize(80, 25));
        doubleSpinBox_6->setFont(font2);
        doubleSpinBox_6->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_6->setDecimals(5);
        doubleSpinBox_6->setMinimum(-99.989999999999995);

        gridLayout_4->addWidget(doubleSpinBox_6, 2, 1, 1, 1);

        label_2 = new QLabel(tab);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setMaximumSize(QSize(20, 25));
        label_2->setFont(font2);

        gridLayout_4->addWidget(label_2, 1, 0, 1, 1);


        gridLayout_3->addLayout(gridLayout_4, 2, 0, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        gridLayout_3->addLayout(verticalLayout, 2, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout_3);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_3);

        QIcon icon23;
        icon23.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/parameters.png"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab, icon23, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        verticalLayout_3 = new QVBoxLayout(tab_3);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        gridLayout_5 = new QGridLayout();
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalSpacer_6 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_6);


        gridLayout_5->addLayout(verticalLayout_4, 2, 1, 1, 1);

        gridLayout_11 = new QGridLayout();
        gridLayout_11->setObjectName(QString::fromUtf8("gridLayout_11"));
        label_7 = new QLabel(tab_3);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setMaximumSize(QSize(20, 25));
        label_7->setFont(font2);

        gridLayout_11->addWidget(label_7, 1, 0, 1, 1);

        label_8 = new QLabel(tab_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setMaximumSize(QSize(20, 25));
        label_8->setFont(font2);

        gridLayout_11->addWidget(label_8, 2, 0, 1, 1);

        doubleSpinBox_2 = new QDoubleSpinBox(tab_3);
        doubleSpinBox_2->setObjectName(QString::fromUtf8("doubleSpinBox_2"));
        doubleSpinBox_2->setMinimumSize(QSize(80, 25));
        doubleSpinBox_2->setMaximumSize(QSize(80, 25));
        doubleSpinBox_2->setFont(font2);
        doubleSpinBox_2->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_2->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_2->setMaximum(10000000000000.000000000000000);

        gridLayout_11->addWidget(doubleSpinBox_2, 1, 1, 1, 1);

        doubleSpinBox_3 = new QDoubleSpinBox(tab_3);
        doubleSpinBox_3->setObjectName(QString::fromUtf8("doubleSpinBox_3"));
        doubleSpinBox_3->setMinimumSize(QSize(80, 25));
        doubleSpinBox_3->setMaximumSize(QSize(80, 25));
        doubleSpinBox_3->setFont(font2);
        doubleSpinBox_3->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_3->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_3->setMaximum(10000000000000.000000000000000);

        gridLayout_11->addWidget(doubleSpinBox_3, 2, 1, 1, 1);

        doubleSpinBox = new QDoubleSpinBox(tab_3);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        doubleSpinBox->setMinimumSize(QSize(80, 25));
        doubleSpinBox->setMaximumSize(QSize(80, 25));
        doubleSpinBox->setFont(font2);
        doubleSpinBox->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox->setMaximum(10000000000000.000000000000000);

        gridLayout_11->addWidget(doubleSpinBox, 0, 1, 1, 1);

        label_6 = new QLabel(tab_3);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setMaximumSize(QSize(20, 25));
        label_6->setFont(font2);

        gridLayout_11->addWidget(label_6, 0, 0, 1, 1);


        gridLayout_5->addLayout(gridLayout_11, 2, 0, 1, 1);

        pushButton_36 = new QPushButton(tab_3);
        pushButton_36->setObjectName(QString::fromUtf8("pushButton_36"));
        pushButton_36->setMinimumSize(QSize(25, 25));
        pushButton_36->setMaximumSize(QSize(25, 25));
        pushButton_36->setFont(font);
        pushButton_36->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"    padding: 2px;\n"
"}"));
        pushButton_36->setIcon(icon22);

        gridLayout_5->addWidget(pushButton_36, 1, 1, 1, 1);

        label_5 = new QLabel(tab_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setMinimumSize(QSize(130, 25));
        label_5->setMaximumSize(QSize(130, 25));
        label_5->setFont(font2);

        gridLayout_5->addWidget(label_5, 0, 0, 1, 1);

        comboBox = new QComboBox(tab_3);
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setMinimumSize(QSize(110, 25));
        comboBox->setMaximumSize(QSize(110, 25));
        comboBox->setFont(font2);
        comboBox->setStyleSheet(QString::fromUtf8("\n"
"padding-left:5px;"));

        gridLayout_5->addWidget(comboBox, 1, 0, 1, 1);


        verticalLayout_3->addLayout(gridLayout_5);

        verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer_5);

        tabWidget->addTab(tab_3, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_6 = new QVBoxLayout(tab_2);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        comboBox_4 = new QComboBox(tab_2);
        comboBox_4->addItem(QString());
        comboBox_4->setObjectName(QString::fromUtf8("comboBox_4"));
        comboBox_4->setFont(font2);

        horizontalLayout->addWidget(comboBox_4);

        pushButton_10 = new QPushButton(tab_2);
        pushButton_10->setObjectName(QString::fromUtf8("pushButton_10"));
        pushButton_10->setMinimumSize(QSize(25, 25));
        pushButton_10->setMaximumSize(QSize(25, 25));
        QIcon icon24;
        icon24.addFile(QString::fromUtf8(":/amwind/figure/am_wind/open.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_10->setIcon(icon24);

        horizontalLayout->addWidget(pushButton_10);

        pushButton_11 = new QPushButton(tab_2);
        pushButton_11->setObjectName(QString::fromUtf8("pushButton_11"));
        pushButton_11->setMinimumSize(QSize(25, 25));
        pushButton_11->setMaximumSize(QSize(25, 25));
        QIcon icon25;
        icon25.addFile(QString::fromUtf8(":/amwind/figure/am_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_11->setIcon(icon25);

        horizontalLayout->addWidget(pushButton_11);


        verticalLayout_6->addLayout(horizontalLayout);

        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        doubleSpinBox_8 = new QDoubleSpinBox(tab_2);
        doubleSpinBox_8->setObjectName(QString::fromUtf8("doubleSpinBox_8"));
        doubleSpinBox_8->setMinimumSize(QSize(70, 25));
        doubleSpinBox_8->setMaximumSize(QSize(70, 25));
        doubleSpinBox_8->setFont(font2);
        doubleSpinBox_8->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_8->setMinimum(-10000000000000000.000000000000000);

        gridLayout_2->addWidget(doubleSpinBox_8, 2, 1, 1, 1);

        doubleSpinBox_7 = new QDoubleSpinBox(tab_2);
        doubleSpinBox_7->setObjectName(QString::fromUtf8("doubleSpinBox_7"));
        doubleSpinBox_7->setMinimumSize(QSize(70, 25));
        doubleSpinBox_7->setMaximumSize(QSize(70, 25));
        doubleSpinBox_7->setFont(font2);
        doubleSpinBox_7->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_7->setMinimum(-10000000000000000.000000000000000);

        gridLayout_2->addWidget(doubleSpinBox_7, 1, 1, 1, 1);

        pushButton_8 = new QPushButton(tab_2);
        pushButton_8->setObjectName(QString::fromUtf8("pushButton_8"));
        pushButton_8->setMinimumSize(QSize(25, 25));
        pushButton_8->setMaximumSize(QSize(25, 25));
        pushButton_8->setStyleSheet(QString::fromUtf8("QPushButton::hover{\n"
"    background-color: transparent;\n"
"    border: 2px solid #6A8480;\n"
"    border-radius: 2px;\n"
"    padding: 2px;\n"
"}"));
        pushButton_8->setIcon(icon22);

        gridLayout_2->addWidget(pushButton_8, 0, 2, 1, 1);

        comboBox_3 = new QComboBox(tab_2);
        comboBox_3->addItem(QString());
        comboBox_3->addItem(QString());
        comboBox_3->setObjectName(QString::fromUtf8("comboBox_3"));
        comboBox_3->setMinimumSize(QSize(70, 25));
        comboBox_3->setMaximumSize(QSize(70, 25));
        comboBox_3->setFont(font2);

        gridLayout_2->addWidget(comboBox_3, 0, 1, 1, 1);

        label_9 = new QLabel(tab_2);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setMinimumSize(QSize(35, 25));
        label_9->setMaximumSize(QSize(35, 25));
        label_9->setFont(font2);

        gridLayout_2->addWidget(label_9, 0, 0, 1, 1);

        label_10 = new QLabel(tab_2);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setMinimumSize(QSize(35, 25));
        label_10->setMaximumSize(QSize(35, 25));
        label_10->setFont(font2);

        gridLayout_2->addWidget(label_10, 1, 0, 1, 1);

        doubleSpinBox_9 = new QDoubleSpinBox(tab_2);
        doubleSpinBox_9->setObjectName(QString::fromUtf8("doubleSpinBox_9"));
        doubleSpinBox_9->setMinimumSize(QSize(70, 25));
        doubleSpinBox_9->setMaximumSize(QSize(70, 25));
        doubleSpinBox_9->setFont(font2);
        doubleSpinBox_9->setStyleSheet(QString::fromUtf8("padding-left:5px;"));
        doubleSpinBox_9->setMinimum(-10000000000000000.000000000000000);

        gridLayout_2->addWidget(doubleSpinBox_9, 3, 1, 1, 1);


        verticalLayout_6->addLayout(gridLayout_2);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_2);

        tabWidget->addTab(tab_2, icon15, QString());

        verticalLayout_5->addWidget(tabWidget);


        retranslateUi(CADDockWidget);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(CADDockWidget);
    } // setupUi

    void retranslateUi(QWidget *CADDockWidget)
    {
        CADDockWidget->setWindowTitle(QApplication::translate("CADDockWidget", "Form", nullptr));
        actionSphere->setText(QApplication::translate("CADDockWidget", "Ball", nullptr));
        actionCube->setText(QApplication::translate("CADDockWidget", "Cube", nullptr));
        actionTorus->setText(QApplication::translate("CADDockWidget", "Torus", nullptr));
        actionCone->setText(QApplication::translate("CADDockWidget", "Cone", nullptr));
        actionCylinder->setText(QApplication::translate("CADDockWidget", "Cylinder", nullptr));
        actionPoint->setText(QApplication::translate("CADDockWidget", "Point", nullptr));
        actionLine->setText(QApplication::translate("CADDockWidget", "Line", nullptr));
        actionSquare->setText(QApplication::translate("CADDockWidget", "Square", nullptr));
        actionUnion->setText(QApplication::translate("CADDockWidget", "Union", nullptr));
        actionSection->setText(QApplication::translate("CADDockWidget", "Section", nullptr));
        actionCut->setText(QApplication::translate("CADDockWidget", "Cut", nullptr));
        actionSweep->setText(QApplication::translate("CADDockWidget", "Sweep", nullptr));
        actionExtrude->setText(QApplication::translate("CADDockWidget", "Extrude", nullptr));
        actionMirror->setText(QApplication::translate("CADDockWidget", "Mirror", nullptr));
        actionSelectFace->setText(QApplication::translate("CADDockWidget", "Face", nullptr));
        actionSelectDomain->setText(QApplication::translate("CADDockWidget", "Domain", nullptr));
#ifndef QT_NO_TOOLTIP
        actionSelectDomain->setToolTip(QApplication::translate("CADDockWidget", "Domain", nullptr));
#endif // QT_NO_TOOLTIP
        actionPart1->setText(QApplication::translate("CADDockWidget", "Part1", nullptr));
        actionPart2->setText(QApplication::translate("CADDockWidget", "Part2", nullptr));
        pushButton_4->setText(QString());
        pushButton_9->setText(QString());
        pushButton->setText(QString());
        pushButton_2->setText(QString());
        pushButton_3->setText(QString());
        pushButton_7->setText(QString());
        pushButton_6->setText(QString());
        pushButton_5->setText(QString());
        pushButton_13->setText(QString());
        pushButton_0->setText(QString());
#ifndef QT_NO_WHATSTHIS
        tabWidget->setWhatsThis(QApplication::translate("CADDockWidget", "<html><head/><body><p>Properties</p><p><br/></p></body></html>", nullptr));
#endif // QT_NO_WHATSTHIS
        pushButton_33->setText(QString());
        label_4->setText(QApplication::translate("CADDockWidget", "Parameters:", nullptr));
        label->setText(QApplication::translate("CADDockWidget", "x0:", nullptr));
        label_3->setText(QApplication::translate("CADDockWidget", "x2:", nullptr));
        label_2->setText(QApplication::translate("CADDockWidget", "x1:", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QString());
        tabWidget->setTabToolTip(tabWidget->indexOf(tab), QApplication::translate("CADDockWidget", "Properties", nullptr));
        label_7->setText(QApplication::translate("CADDockWidget", "x1:", nullptr));
        label_8->setText(QApplication::translate("CADDockWidget", "x2:", nullptr));
        label_6->setText(QApplication::translate("CADDockWidget", "x0:", nullptr));
        pushButton_36->setText(QString());
        label_5->setText(QApplication::translate("CADDockWidget", "Parameters:", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QString());
        tabWidget->setTabToolTip(tabWidget->indexOf(tab_3), QApplication::translate("CADDockWidget", "Other Operations", nullptr));
        comboBox_4->setItemText(0, QApplication::translate("CADDockWidget", "J2 plasticity", nullptr));

        pushButton_10->setText(QString());
        pushButton_11->setText(QString());
        pushButton_8->setText(QString());
        comboBox_3->setItemText(0, QApplication::translate("CADDockWidget", "Dirichlet", nullptr));
        comboBox_3->setItemText(1, QApplication::translate("CADDockWidget", "Neumann", nullptr));

        label_9->setText(QApplication::translate("CADDockWidget", "Type:", nullptr));
        label_10->setText(QApplication::translate("CADDockWidget", "Value:", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QString());
    } // retranslateUi

};

namespace Ui {
    class CADDockWidget: public Ui_CADDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CADDOCKWIDGET_H
