/********************************************************************************
** Form generated from reading UI file 'MeasureDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MEASUREDOCKWIDGET_H
#define UI_MEASUREDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MeasureDockWidget
{
public:
    QAction *actionStraightness;
    QAction *actionFlatness;
    QAction *actionSurface;
    QAction *actionLine;
    QAction *actionCircularity;
    QAction *actionCylindricity;
    QAction *actionLineProfile;
    QAction *actionSurfaceProfile;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;
    QSpacerItem *horizontalSpacer_8;
    QPushButton *pushButton_3;
    QPushButton *pushButton_6;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_4;
    QSpacerItem *horizontalSpacer_3;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QSpacerItem *horizontalSpacer_4;
    QDoubleSpinBox *doubleSpinBox;
    QSpacerItem *horizontalSpacer_9;
    QPushButton *pushButton_5;
    QScrollBar *horizontalScrollBar;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_9;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_8;
    QDoubleSpinBox *doubleSpinBox_2;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *pushButton_4;
    QPushButton *pushButton_7;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_5;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton_8;
    QPushButton *pushButton_9;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_6;
    QSpacerItem *horizontalSpacer_6;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_11;
    QDoubleSpinBox *doubleSpinBox_3;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *pushButton_19;
    QPushButton *pushButton_11;
    QProgressBar *progressBar;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QPushButton *pushButton_10;
    QPushButton *pushButton_13;
    QProgressBar *progressBar_2;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *MeasureDockWidget)
    {
        if (MeasureDockWidget->objectName().isEmpty())
            MeasureDockWidget->setObjectName(QString::fromUtf8("MeasureDockWidget"));
        MeasureDockWidget->resize(315, 584);
        actionStraightness = new QAction(MeasureDockWidget);
        actionStraightness->setObjectName(QString::fromUtf8("actionStraightness"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/straight.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStraightness->setIcon(icon);
        QFont font;
        font.setPointSize(9);
        actionStraightness->setFont(font);
        actionFlatness = new QAction(MeasureDockWidget);
        actionFlatness->setObjectName(QString::fromUtf8("actionFlatness"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/flatness.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionFlatness->setIcon(icon1);
        actionFlatness->setFont(font);
        actionSurface = new QAction(MeasureDockWidget);
        actionSurface->setObjectName(QString::fromUtf8("actionSurface"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/face.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSurface->setIcon(icon2);
        actionSurface->setFont(font);
        actionLine = new QAction(MeasureDockWidget);
        actionLine->setObjectName(QString::fromUtf8("actionLine"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/line.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionLine->setIcon(icon3);
        actionLine->setFont(font);
        actionCircularity = new QAction(MeasureDockWidget);
        actionCircularity->setObjectName(QString::fromUtf8("actionCircularity"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/circularity.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCircularity->setIcon(icon4);
        actionCircularity->setFont(font);
        actionCylindricity = new QAction(MeasureDockWidget);
        actionCylindricity->setObjectName(QString::fromUtf8("actionCylindricity"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/cylindricity.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCylindricity->setIcon(icon5);
        actionCylindricity->setFont(font);
        actionLineProfile = new QAction(MeasureDockWidget);
        actionLineProfile->setObjectName(QString::fromUtf8("actionLineProfile"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/lineprofile.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionLineProfile->setIcon(icon6);
        actionLineProfile->setFont(font);
        actionSurfaceProfile = new QAction(MeasureDockWidget);
        actionSurfaceProfile->setObjectName(QString::fromUtf8("actionSurfaceProfile"));
        actionSurfaceProfile->setCheckable(false);
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/surfaceprofile.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSurfaceProfile->setIcon(icon7);
        actionSurfaceProfile->setFont(font);
        verticalLayout = new QVBoxLayout(MeasureDockWidget);
        verticalLayout->setSpacing(5);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label = new QLabel(MeasureDockWidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFont(font);

        horizontalLayout_3->addWidget(label);

        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_8);

        pushButton_3 = new QPushButton(MeasureDockWidget);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        pushButton_3->setMinimumSize(QSize(25, 25));
        pushButton_3->setMaximumSize(QSize(25, 25));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/open.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_3->setIcon(icon8);

        horizontalLayout_3->addWidget(pushButton_3);

        pushButton_6 = new QPushButton(MeasureDockWidget);
        pushButton_6->setObjectName(QString::fromUtf8("pushButton_6"));
        pushButton_6->setMinimumSize(QSize(25, 25));
        pushButton_6->setMaximumSize(QSize(25, 25));
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/view.png"), QSize(), QIcon::Normal, QIcon::Off);
        icon9.addFile(QString::fromUtf8(":/new/measure/figure/cad_wind/noview.png"), QSize(), QIcon::Normal, QIcon::On);
        icon9.addFile(QString::fromUtf8(":/new/measure/figure/cad_wind/noview.png"), QSize(), QIcon::Selected, QIcon::Off);
        icon9.addFile(QString::fromUtf8(":/new/measure/figure/measure_wind/hide.png"), QSize(), QIcon::Selected, QIcon::On);
        pushButton_6->setIcon(icon9);
        pushButton_6->setCheckable(true);
        pushButton_6->setChecked(false);

        horizontalLayout_3->addWidget(pushButton_6);


        verticalLayout->addLayout(horizontalLayout_3);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_4 = new QLabel(MeasureDockWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFont(font);

        horizontalLayout_7->addWidget(label_4);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_3);


        verticalLayout->addLayout(horizontalLayout_7);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        pushButton = new QPushButton(MeasureDockWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setMinimumSize(QSize(25, 25));
        pushButton->setMaximumSize(QSize(25, 25));
        pushButton->setIcon(icon2);
        pushButton->setIconSize(QSize(16, 16));
        pushButton->setCheckable(true);

        horizontalLayout_4->addWidget(pushButton);

        pushButton_2 = new QPushButton(MeasureDockWidget);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setEnabled(true);
        pushButton_2->setMinimumSize(QSize(25, 25));
        pushButton_2->setMaximumSize(QSize(25, 25));
        pushButton_2->setIcon(icon7);
        pushButton_2->setIconSize(QSize(16, 16));
        pushButton_2->setCheckable(true);
        pushButton_2->setChecked(false);

        horizontalLayout_4->addWidget(pushButton_2);

        horizontalSpacer_4 = new QSpacerItem(3, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_4);

        doubleSpinBox = new QDoubleSpinBox(MeasureDockWidget);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        doubleSpinBox->setMinimumSize(QSize(60, 25));
        doubleSpinBox->setMaximumSize(QSize(60, 25));
        doubleSpinBox->setFont(font);
        doubleSpinBox->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox->setDecimals(3);
        doubleSpinBox->setSingleStep(0.010000000000000);
        doubleSpinBox->setValue(0.050000000000000);

        horizontalLayout_4->addWidget(doubleSpinBox);

        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_9);

        pushButton_5 = new QPushButton(MeasureDockWidget);
        pushButton_5->setObjectName(QString::fromUtf8("pushButton_5"));
        pushButton_5->setMinimumSize(QSize(25, 25));
        pushButton_5->setMaximumSize(QSize(25, 25));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/view.png"), QSize(), QIcon::Normal, QIcon::Off);
        icon10.addFile(QString::fromUtf8(":/new/measure/figure/cad_wind/noview.png"), QSize(), QIcon::Normal, QIcon::On);
        pushButton_5->setIcon(icon10);
        pushButton_5->setCheckable(true);

        horizontalLayout_4->addWidget(pushButton_5);


        verticalLayout->addLayout(horizontalLayout_4);

        horizontalScrollBar = new QScrollBar(MeasureDockWidget);
        horizontalScrollBar->setObjectName(QString::fromUtf8("horizontalScrollBar"));
        horizontalScrollBar->setMinimum(0);
        horizontalScrollBar->setMaximum(100);
        horizontalScrollBar->setValue(100);
        horizontalScrollBar->setOrientation(Qt::Horizontal);

        verticalLayout->addWidget(horizontalScrollBar);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        label_9 = new QLabel(MeasureDockWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setMinimumSize(QSize(0, 25));
        label_9->setMaximumSize(QSize(16777215, 25));
        label_9->setFont(font);

        verticalLayout_2->addWidget(label_9);


        verticalLayout->addLayout(verticalLayout_2);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setSpacing(0);
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label_8 = new QLabel(MeasureDockWidget);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setFont(font);

        horizontalLayout_12->addWidget(label_8);

        doubleSpinBox_2 = new QDoubleSpinBox(MeasureDockWidget);
        doubleSpinBox_2->setObjectName(QString::fromUtf8("doubleSpinBox_2"));
        doubleSpinBox_2->setMinimumSize(QSize(60, 25));
        doubleSpinBox_2->setMaximumSize(QSize(60, 25));
        doubleSpinBox_2->setFont(font);
        doubleSpinBox_2->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox_2->setMinimum(0.010000000000000);
        doubleSpinBox_2->setValue(1.000000000000000);

        horizontalLayout_12->addWidget(doubleSpinBox_2);

        horizontalSpacer_5 = new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_12->addItem(horizontalSpacer_5);

        pushButton_4 = new QPushButton(MeasureDockWidget);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));
        pushButton_4->setMinimumSize(QSize(25, 25));
        pushButton_4->setMaximumSize(QSize(25, 25));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/cad_wind/figure/cad_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_4->setIcon(icon11);

        horizontalLayout_12->addWidget(pushButton_4);

        pushButton_7 = new QPushButton(MeasureDockWidget);
        pushButton_7->setObjectName(QString::fromUtf8("pushButton_7"));
        pushButton_7->setMinimumSize(QSize(25, 25));
        pushButton_7->setMaximumSize(QSize(25, 25));
        pushButton_7->setIcon(icon10);
        pushButton_7->setCheckable(true);

        horizontalLayout_12->addWidget(pushButton_7);


        verticalLayout->addLayout(horizontalLayout_12);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setSpacing(0);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_5 = new QLabel(MeasureDockWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setFont(font);

        horizontalLayout_10->addWidget(label_5);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer);

        pushButton_8 = new QPushButton(MeasureDockWidget);
        pushButton_8->setObjectName(QString::fromUtf8("pushButton_8"));
        pushButton_8->setMinimumSize(QSize(25, 25));
        pushButton_8->setMaximumSize(QSize(25, 25));
        pushButton_8->setIcon(icon8);

        horizontalLayout_10->addWidget(pushButton_8);

        pushButton_9 = new QPushButton(MeasureDockWidget);
        pushButton_9->setObjectName(QString::fromUtf8("pushButton_9"));
        pushButton_9->setMinimumSize(QSize(25, 25));
        pushButton_9->setMaximumSize(QSize(25, 25));
        pushButton_9->setIcon(icon10);
        pushButton_9->setCheckable(true);

        horizontalLayout_10->addWidget(pushButton_9);


        verticalLayout->addLayout(horizontalLayout_10);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        label_6 = new QLabel(MeasureDockWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setMinimumSize(QSize(0, 25));
        label_6->setMaximumSize(QSize(16777215, 25));
        label_6->setFont(font);

        horizontalLayout_11->addWidget(label_6);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_6);


        verticalLayout->addLayout(horizontalLayout_11);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(0);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_11 = new QLabel(MeasureDockWidget);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setFont(font);
        label_11->setMargin(0);

        horizontalLayout_8->addWidget(label_11);

        doubleSpinBox_3 = new QDoubleSpinBox(MeasureDockWidget);
        doubleSpinBox_3->setObjectName(QString::fromUtf8("doubleSpinBox_3"));
        doubleSpinBox_3->setMinimumSize(QSize(60, 25));
        doubleSpinBox_3->setMaximumSize(QSize(16777215, 25));
        doubleSpinBox_3->setFont(font);
        doubleSpinBox_3->setStyleSheet(QString::fromUtf8("padding-left:3px;"));
        doubleSpinBox_3->setMinimum(0.000000000000000);
        doubleSpinBox_3->setValue(1.000000000000000);

        horizontalLayout_8->addWidget(doubleSpinBox_3);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_2);

        pushButton_19 = new QPushButton(MeasureDockWidget);
        pushButton_19->setObjectName(QString::fromUtf8("pushButton_19"));
        pushButton_19->setMinimumSize(QSize(25, 25));
        pushButton_19->setMaximumSize(QSize(25, 25));
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/amwind/figure/am_wind/ok.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_19->setIcon(icon12);

        horizontalLayout_8->addWidget(pushButton_19);

        pushButton_11 = new QPushButton(MeasureDockWidget);
        pushButton_11->setObjectName(QString::fromUtf8("pushButton_11"));
        pushButton_11->setMinimumSize(QSize(25, 25));
        pushButton_11->setMaximumSize(QSize(25, 25));
        pushButton_11->setIcon(icon10);
        pushButton_11->setCheckable(true);

        horizontalLayout_8->addWidget(pushButton_11);


        verticalLayout->addLayout(horizontalLayout_8);

        progressBar = new QProgressBar(MeasureDockWidget);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setSizeIncrement(QSize(0, 25));
        progressBar->setBaseSize(QSize(0, 25));
        progressBar->setFont(font);
        progressBar->setValue(0);
        progressBar->setTextVisible(true);

        verticalLayout->addWidget(progressBar);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(MeasureDockWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);

        horizontalLayout->addWidget(label_2);

        pushButton_10 = new QPushButton(MeasureDockWidget);
        pushButton_10->setObjectName(QString::fromUtf8("pushButton_10"));
        pushButton_10->setMinimumSize(QSize(25, 25));
        pushButton_10->setMaximumSize(QSize(25, 25));
        pushButton_10->setIcon(icon12);

        horizontalLayout->addWidget(pushButton_10);

        pushButton_13 = new QPushButton(MeasureDockWidget);
        pushButton_13->setObjectName(QString::fromUtf8("pushButton_13"));
        pushButton_13->setMinimumSize(QSize(25, 25));
        pushButton_13->setMaximumSize(QSize(25, 25));
        pushButton_13->setIcon(icon10);
        pushButton_13->setCheckable(true);

        horizontalLayout->addWidget(pushButton_13);


        verticalLayout->addLayout(horizontalLayout);

        progressBar_2 = new QProgressBar(MeasureDockWidget);
        progressBar_2->setObjectName(QString::fromUtf8("progressBar_2"));
        progressBar_2->setSizeIncrement(QSize(0, 25));
        progressBar_2->setBaseSize(QSize(0, 25));
        progressBar_2->setFont(font);
        progressBar_2->setValue(0);

        verticalLayout->addWidget(progressBar_2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(MeasureDockWidget);

        QMetaObject::connectSlotsByName(MeasureDockWidget);
    } // setupUi

    void retranslateUi(QWidget *MeasureDockWidget)
    {
        MeasureDockWidget->setWindowTitle(QApplication::translate("MeasureDockWidget", "Form", nullptr));
        actionStraightness->setText(QApplication::translate("MeasureDockWidget", "straight", nullptr));
#ifndef QT_NO_TOOLTIP
        actionStraightness->setToolTip(QApplication::translate("MeasureDockWidget", "Straightness", nullptr));
#endif // QT_NO_TOOLTIP
        actionFlatness->setText(QApplication::translate("MeasureDockWidget", "flat", nullptr));
#ifndef QT_NO_TOOLTIP
        actionFlatness->setToolTip(QApplication::translate("MeasureDockWidget", "Flatness", nullptr));
#endif // QT_NO_TOOLTIP
        actionSurface->setText(QApplication::translate("MeasureDockWidget", "face", nullptr));
        actionLine->setText(QApplication::translate("MeasureDockWidget", "edge", nullptr));
        actionCircularity->setText(QApplication::translate("MeasureDockWidget", "Circularity", nullptr));
        actionCylindricity->setText(QApplication::translate("MeasureDockWidget", "Cylindricity", nullptr));
#ifndef QT_NO_TOOLTIP
        actionCylindricity->setToolTip(QApplication::translate("MeasureDockWidget", "Cylindricity", nullptr));
#endif // QT_NO_TOOLTIP
        actionLineProfile->setText(QApplication::translate("MeasureDockWidget", "line", nullptr));
#ifndef QT_NO_TOOLTIP
        actionLineProfile->setToolTip(QApplication::translate("MeasureDockWidget", "line", nullptr));
#endif // QT_NO_TOOLTIP
        actionSurfaceProfile->setText(QApplication::translate("MeasureDockWidget", "surface", nullptr));
        label->setText(QApplication::translate("MeasureDockWidget", "1. Import CAD", nullptr));
        pushButton_3->setText(QString());
        pushButton_6->setText(QString());
        label_4->setText(QApplication::translate("MeasureDockWidget", "2. Set GDT", nullptr));
        pushButton->setText(QString());
        pushButton_2->setText(QString());
        pushButton_5->setText(QString());
        label_9->setText(QApplication::translate("MeasureDockWidget", "3. CAD to Point Cloud", nullptr));
        label_8->setText(QApplication::translate("MeasureDockWidget", "Dens:  ", nullptr));
        pushButton_4->setText(QString());
        pushButton_7->setText(QString());
        label_5->setText(QApplication::translate("MeasureDockWidget", "4. Import PC", nullptr));
        pushButton_8->setText(QString());
        pushButton_9->setText(QString());
        label_6->setText(QApplication::translate("MeasureDockWidget", "5. Registration", nullptr));
        label_11->setText(QApplication::translate("MeasureDockWidget", "US:  ", nullptr));
        pushButton_19->setText(QString());
        pushButton_11->setText(QString());
        label_2->setText(QApplication::translate("MeasureDockWidget", "6. Results", nullptr));
        pushButton_10->setText(QString());
        pushButton_13->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class MeasureDockWidget: public Ui_MeasureDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MEASUREDOCKWIDGET_H
