/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMdiArea>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "CAD/OCCWidget.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionCAD;
    QAction *actionMesh;
    QAction *actionFEM;
    QAction *actionVisual;
    QAction *actionAbout;
    QAction *actionSave;
    QAction *actionOpen;
    QAction *actionSPC;
    QAction *actionAxo;
    QAction *actionFront;
    QAction *actionBack;
    QAction *actionLeft;
    QAction *actionRight;
    QAction *actionTop;
    QAction *actionBottom;
    QAction *actionFit;
    QAction *actionNew;
    QAction *actionMeasure;
    QAction *actionSystem;
    QAction *actionAdditiveManufacturing;
    QAction *actionViewRotationH;
    QAction *actionViewRotationV;
    QAction *actionMachining;
    QAction *actionTransport;
    OCCWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QMdiArea *mdiArea;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuFENGSim;
    QMenu *menuView;
    QMenu *menuAbout;
    QStatusBar *statusBar;
    QToolBar *toolBarFENGSim;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_2;
    QPushButton *pushButton;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 637);
        MainWindow->setMinimumSize(QSize(800, 637));
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/Fengsim_logo_about.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        MainWindow->setWindowOpacity(1.000000000000000);
        MainWindow->setStyleSheet(QString::fromUtf8("QWindow {height: 600; width: 800;}"));
        actionCAD = new QAction(MainWindow);
        actionCAD->setObjectName(QString::fromUtf8("actionCAD"));
        actionCAD->setCheckable(true);
        actionCAD->setEnabled(true);
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/cad.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCAD->setIcon(icon1);
        actionMesh = new QAction(MainWindow);
        actionMesh->setObjectName(QString::fromUtf8("actionMesh"));
        actionMesh->setCheckable(true);
        actionMesh->setEnabled(true);
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/mesh.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionMesh->setIcon(icon2);
        actionFEM = new QAction(MainWindow);
        actionFEM->setObjectName(QString::fromUtf8("actionFEM"));
        actionFEM->setCheckable(true);
        actionFEM->setEnabled(true);
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/fem.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionFEM->setIcon(icon3);
        actionVisual = new QAction(MainWindow);
        actionVisual->setObjectName(QString::fromUtf8("actionVisual"));
        actionVisual->setCheckable(true);
        actionVisual->setEnabled(true);
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/visual.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionVisual->setIcon(icon4);
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        actionAbout->setIcon(icon);
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/save.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSave->setIcon(icon5);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/open.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpen->setIcon(icon6);
        actionSPC = new QAction(MainWindow);
        actionSPC->setObjectName(QString::fromUtf8("actionSPC"));
        actionSPC->setCheckable(true);
        actionSPC->setEnabled(true);
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/new/prefix1/figure/spc_wind/spc.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSPC->setIcon(icon7);
        actionAxo = new QAction(MainWindow);
        actionAxo->setObjectName(QString::fromUtf8("actionAxo"));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/axo.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAxo->setIcon(icon8);
        actionAxo->setFont(font);
        actionFront = new QAction(MainWindow);
        actionFront->setObjectName(QString::fromUtf8("actionFront"));
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/front.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionFront->setIcon(icon9);
        actionFront->setFont(font);
        actionBack = new QAction(MainWindow);
        actionBack->setObjectName(QString::fromUtf8("actionBack"));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/back.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionBack->setIcon(icon10);
        actionBack->setFont(font);
        actionLeft = new QAction(MainWindow);
        actionLeft->setObjectName(QString::fromUtf8("actionLeft"));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/left.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionLeft->setIcon(icon11);
        actionLeft->setFont(font);
        actionRight = new QAction(MainWindow);
        actionRight->setObjectName(QString::fromUtf8("actionRight"));
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/right.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionRight->setIcon(icon12);
        actionRight->setFont(font);
        actionTop = new QAction(MainWindow);
        actionTop->setObjectName(QString::fromUtf8("actionTop"));
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/top.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTop->setIcon(icon13);
        actionTop->setFont(font);
        actionBottom = new QAction(MainWindow);
        actionBottom->setObjectName(QString::fromUtf8("actionBottom"));
        QIcon icon14;
        icon14.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/bottom.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionBottom->setIcon(icon14);
        actionBottom->setFont(font);
        actionFit = new QAction(MainWindow);
        actionFit->setObjectName(QString::fromUtf8("actionFit"));
        QIcon icon15;
        icon15.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/fit.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionFit->setIcon(icon15);
        actionFit->setFont(font);
        actionNew = new QAction(MainWindow);
        actionNew->setObjectName(QString::fromUtf8("actionNew"));
        QIcon icon16;
        icon16.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/new.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionNew->setIcon(icon16);
        actionMeasure = new QAction(MainWindow);
        actionMeasure->setObjectName(QString::fromUtf8("actionMeasure"));
        actionMeasure->setCheckable(true);
        actionMeasure->setEnabled(true);
        QIcon icon17;
        icon17.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/measure.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionMeasure->setIcon(icon17);
        actionSystem = new QAction(MainWindow);
        actionSystem->setObjectName(QString::fromUtf8("actionSystem"));
        actionSystem->setCheckable(true);
        actionSystem->setEnabled(true);
        QIcon icon18;
        icon18.addFile(QString::fromUtf8(":/new/prefix1/figure/system_wind/system.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSystem->setIcon(icon18);
        actionAdditiveManufacturing = new QAction(MainWindow);
        actionAdditiveManufacturing->setObjectName(QString::fromUtf8("actionAdditiveManufacturing"));
        actionAdditiveManufacturing->setCheckable(true);
        actionAdditiveManufacturing->setEnabled(true);
        QIcon icon19;
        icon19.addFile(QString::fromUtf8(":/new/prefix1/figure/mesh_wind/am.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAdditiveManufacturing->setIcon(icon19);
        actionViewRotationH = new QAction(MainWindow);
        actionViewRotationH->setObjectName(QString::fromUtf8("actionViewRotationH"));
        actionViewRotationH->setCheckable(true);
        QIcon icon20;
        icon20.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/viewrotationh.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionViewRotationH->setIcon(icon20);
        actionViewRotationV = new QAction(MainWindow);
        actionViewRotationV->setObjectName(QString::fromUtf8("actionViewRotationV"));
        actionViewRotationV->setCheckable(true);
        QIcon icon21;
        icon21.addFile(QString::fromUtf8(":/main_wind/figure/main_wind/viewrotationv.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionViewRotationV->setIcon(icon21);
        actionMachining = new QAction(MainWindow);
        actionMachining->setObjectName(QString::fromUtf8("actionMachining"));
        actionMachining->setCheckable(true);
        actionMachining->setEnabled(true);
        QIcon icon22;
        icon22.addFile(QString::fromUtf8(":/figure/machining/machining.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionMachining->setIcon(icon22);
        actionTransport = new QAction(MainWindow);
        actionTransport->setObjectName(QString::fromUtf8("actionTransport"));
        actionTransport->setCheckable(true);
        QIcon icon23;
        icon23.addFile(QString::fromUtf8(":/new/prefix1/nuclear.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTransport->setIcon(icon23);
        centralWidget = new OCCWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        centralWidget->setStyleSheet(QString::fromUtf8("margin: 0px;"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        mdiArea = new QMdiArea(centralWidget);
        mdiArea->setObjectName(QString::fromUtf8("mdiArea"));
        mdiArea->setEnabled(false);
        mdiArea->setAutoFillBackground(true);
        mdiArea->setStyleSheet(QString::fromUtf8("margin: 0px; padding: 0px;"));
        mdiArea->setLineWidth(0);
        QBrush brush(QColor(202, 201, 200, 255));
        brush.setStyle(Qt::SolidPattern);
        mdiArea->setBackground(brush);

        verticalLayout->addWidget(mdiArea);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 22));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(menuBar->sizePolicy().hasHeightForWidth());
        menuBar->setSizePolicy(sizePolicy);
        menuBar->setFocusPolicy(Qt::NoFocus);
        menuBar->setAutoFillBackground(true);
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuFENGSim = new QMenu(menuBar);
        menuFENGSim->setObjectName(QString::fromUtf8("menuFENGSim"));
        menuView = new QMenu(menuFENGSim);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        menuView->setFont(font);
        QIcon icon24;
        icon24.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/view.png"), QSize(), QIcon::Normal, QIcon::Off);
        menuView->setIcon(icon24);
        menuAbout = new QMenu(menuBar);
        menuAbout->setObjectName(QString::fromUtf8("menuAbout"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);
        toolBarFENGSim = new QToolBar(MainWindow);
        toolBarFENGSim->setObjectName(QString::fromUtf8("toolBarFENGSim"));
        toolBarFENGSim->setEnabled(true);
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(toolBarFENGSim->sizePolicy().hasHeightForWidth());
        toolBarFENGSim->setSizePolicy(sizePolicy1);
        toolBarFENGSim->setMovable(false);
        toolBarFENGSim->setAllowedAreas(Qt::TopToolBarArea);
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBarFENGSim);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        dockWidget->setMinimumSize(QSize(180, 420));
        dockWidget->setMaximumSize(QSize(180, 420));
        dockWidget->setFont(font);
        dockWidget->setFloating(true);
        dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
        dockWidget->setAllowedAreas(Qt::NoDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        dockWidgetContents->setStyleSheet(QString::fromUtf8(""));
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setSizeConstraint(QLayout::SetNoConstraint);
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        pushButton = new QPushButton(dockWidgetContents);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setStyleSheet(QString::fromUtf8("QPushButton{background-color:transparent;\n"
"                               max-width: 32px; max-height: 32px; min-width: 32px; min-height: 32px; border: 0px solid white;\n"
"    border-radius: 2px; margin: 2px; padding: 3px;} \n"
"QPushButton:hover, QPushButton::menu-button:hover { \n"
"                                                background-color: transparent; \n"
"border: 2px solid #6A8480; \n"
"}"));
        pushButton->setIcon(icon24);
        pushButton->setIconSize(QSize(26, 26));

        verticalLayout_2->addWidget(pushButton);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);
        dockWidget->raise();

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuFENGSim->menuAction());
        menuBar->addAction(menuAbout->menuAction());
        menuFile->addAction(actionNew);
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionSave);
        menuFENGSim->addAction(actionCAD);
        menuFENGSim->addAction(actionMesh);
        menuFENGSim->addAction(actionFEM);
        menuFENGSim->addAction(actionVisual);
        menuFENGSim->addAction(menuView->menuAction());
        menuFENGSim->addSeparator();
        menuFENGSim->addAction(actionAdditiveManufacturing);
        menuFENGSim->addAction(actionMeasure);
        menuFENGSim->addAction(actionSystem);
        menuFENGSim->addAction(actionSPC);
        menuView->addAction(actionAxo);
        menuView->addAction(actionFront);
        menuView->addAction(actionBack);
        menuView->addAction(actionLeft);
        menuView->addAction(actionRight);
        menuView->addAction(actionTop);
        menuView->addAction(actionBottom);
        menuView->addAction(actionFit);
        menuAbout->addAction(actionAbout);
        toolBarFENGSim->addAction(actionNew);
        toolBarFENGSim->addAction(actionOpen);
        toolBarFENGSim->addAction(actionSave);
        toolBarFENGSim->addSeparator();
        toolBarFENGSim->addAction(actionCAD);
        toolBarFENGSim->addAction(actionMesh);
        toolBarFENGSim->addAction(actionFEM);
        toolBarFENGSim->addAction(actionVisual);
        toolBarFENGSim->addAction(actionAdditiveManufacturing);
        toolBarFENGSim->addAction(actionMeasure);
        toolBarFENGSim->addAction(actionMachining);
        toolBarFENGSim->addAction(actionSystem);
        toolBarFENGSim->addAction(actionSPC);
        toolBarFENGSim->addAction(actionTransport);
        toolBarFENGSim->addSeparator();
        toolBarFENGSim->addAction(actionAbout);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "FENGSim", nullptr));
        actionCAD->setText(QApplication::translate("MainWindow", "CAD", nullptr));
        actionMesh->setText(QApplication::translate("MainWindow", "Mesh", nullptr));
        actionFEM->setText(QApplication::translate("MainWindow", "FEM", nullptr));
        actionVisual->setText(QApplication::translate("MainWindow", "Visual", nullptr));
        actionAbout->setText(QApplication::translate("MainWindow", "About", nullptr));
        actionSave->setText(QApplication::translate("MainWindow", "Save", nullptr));
        actionOpen->setText(QApplication::translate("MainWindow", "Open", nullptr));
        actionSPC->setText(QApplication::translate("MainWindow", "SPC", nullptr));
#ifndef QT_NO_TOOLTIP
        actionSPC->setToolTip(QApplication::translate("MainWindow", "Statistical Process Control", nullptr));
#endif // QT_NO_TOOLTIP
        actionAxo->setText(QApplication::translate("MainWindow", "Axo", nullptr));
        actionFront->setText(QApplication::translate("MainWindow", "Front", nullptr));
        actionBack->setText(QApplication::translate("MainWindow", "Back", nullptr));
        actionLeft->setText(QApplication::translate("MainWindow", "Left", nullptr));
        actionRight->setText(QApplication::translate("MainWindow", "Right", nullptr));
        actionTop->setText(QApplication::translate("MainWindow", "Top", nullptr));
        actionBottom->setText(QApplication::translate("MainWindow", "Bottom", nullptr));
        actionFit->setText(QApplication::translate("MainWindow", "Fit", nullptr));
        actionNew->setText(QApplication::translate("MainWindow", "New", nullptr));
        actionMeasure->setText(QApplication::translate("MainWindow", "Measure", nullptr));
        actionSystem->setText(QApplication::translate("MainWindow", "System", nullptr));
        actionAdditiveManufacturing->setText(QApplication::translate("MainWindow", "AM", nullptr));
#ifndef QT_NO_TOOLTIP
        actionAdditiveManufacturing->setToolTip(QApplication::translate("MainWindow", "AM", nullptr));
#endif // QT_NO_TOOLTIP
        actionViewRotationH->setText(QApplication::translate("MainWindow", "Horizontal", nullptr));
#ifndef QT_NO_TOOLTIP
        actionViewRotationH->setToolTip(QApplication::translate("MainWindow", "ViewRotationH", nullptr));
#endif // QT_NO_TOOLTIP
        actionViewRotationV->setText(QApplication::translate("MainWindow", " Vertical", nullptr));
        actionMachining->setText(QApplication::translate("MainWindow", "Machining", nullptr));
        actionTransport->setText(QApplication::translate("MainWindow", "Transport", nullptr));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", nullptr));
        menuFENGSim->setTitle(QApplication::translate("MainWindow", "FENGSim", nullptr));
        menuView->setTitle(QApplication::translate("MainWindow", "View", nullptr));
        menuAbout->setTitle(QApplication::translate("MainWindow", "Help", nullptr));
        toolBarFENGSim->setWindowTitle(QApplication::translate("MainWindow", "toolBar_3", nullptr));
        pushButton->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
