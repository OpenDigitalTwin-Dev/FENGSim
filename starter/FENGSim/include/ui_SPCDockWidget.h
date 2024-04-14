/********************************************************************************
** Form generated from reading UI file 'SPCDockWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.10
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SPCDOCKWIDGET_H
#define UI_SPCDOCKWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SPCDockWidget
{
public:
    QHBoxLayout *horizontalLayout;
    QTreeWidget *treeWidget;

    void setupUi(QWidget *SPCDockWidget)
    {
        if (SPCDockWidget->objectName().isEmpty())
            SPCDockWidget->setObjectName(QString::fromUtf8("SPCDockWidget"));
        SPCDockWidget->resize(400, 300);
        horizontalLayout = new QHBoxLayout(SPCDockWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        treeWidget = new QTreeWidget(SPCDockWidget);
        QFont font;
        font.setPointSize(9);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setFont(0, font);
        treeWidget->setHeaderItem(__qtreewidgetitem);
        QTreeWidgetItem *__qtreewidgetitem1 = new QTreeWidgetItem(treeWidget);
        new QTreeWidgetItem(__qtreewidgetitem1);
        treeWidget->setObjectName(QString::fromUtf8("treeWidget"));
        treeWidget->setFont(font);
        treeWidget->setStyleSheet(QString::fromUtf8("QTreeView::item:hover {\n"
"    background: transparent;\n"
"}\n"
"QTreeView::item:selected{\n"
"    background: #1E90FF;\n"
"}\n"
"QTreeView::branch {\n"
"    background: transparent;\n"
"}\n"
"QTreeView::branch:hover {\n"
"    background: transparent;\n"
"}\n"
"QTreeView::branch:selected {\n"
"    background: #1E90FF;\n"
"}\n"
"QTreeView::branch:closed:has-children{\n"
"    image: url(:/new/prefix1/figure/spc_wind/open.png);\n"
"}\n"
"QTreeView::branch:open:has-children{\n"
"	image: url(:/new/prefix1/figure/spc_wind/close.png);\n"
"}\n"
""));

        horizontalLayout->addWidget(treeWidget);


        retranslateUi(SPCDockWidget);

        QMetaObject::connectSlotsByName(SPCDockWidget);
    } // setupUi

    void retranslateUi(QWidget *SPCDockWidget)
    {
        SPCDockWidget->setWindowTitle(QApplication::translate("SPCDockWidget", "Form", nullptr));
        QTreeWidgetItem *___qtreewidgetitem = treeWidget->headerItem();
        ___qtreewidgetitem->setText(0, QApplication::translate("SPCDockWidget", "part_1", nullptr));

        const bool __sortingEnabled = treeWidget->isSortingEnabled();
        treeWidget->setSortingEnabled(false);
        QTreeWidgetItem *___qtreewidgetitem1 = treeWidget->topLevelItem(0);
        ___qtreewidgetitem1->setText(0, QApplication::translate("SPCDockWidget", "tool_1", nullptr));
        QTreeWidgetItem *___qtreewidgetitem2 = ___qtreewidgetitem1->child(0);
        ___qtreewidgetitem2->setText(0, QApplication::translate("SPCDockWidget", "data_1", nullptr));
        treeWidget->setSortingEnabled(__sortingEnabled);

    } // retranslateUi

};

namespace Ui {
    class SPCDockWidget: public Ui_SPCDockWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SPCDOCKWIDGET_H
