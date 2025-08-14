/********************************************************************************
** Form generated from reading UI file 'exitdialog.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_EXITDIALOG_H
#define UI_EXITDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_exitDialog
{
public:
    QLabel *label;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton;

    void setupUi(QDialog *exitDialog)
    {
        if (exitDialog->objectName().isEmpty())
            exitDialog->setObjectName(QString::fromUtf8("exitDialog"));
        exitDialog->resize(205, 75);
        exitDialog->setStyleSheet(QString::fromUtf8("QDialog#exitDialog{\n"
"	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(111, 111, 111, 255), stop:1 rgba(255, 255, 255, 255));\n"
"}"));
        label = new QLabel(exitDialog);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(10, 10, 181, 21));
        QFont font;
        font.setBold(true);
        font.setItalic(true);
        font.setWeight(75);
        label->setFont(font);
        layoutWidget = new QWidget(exitDialog);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(10, 44, 186, 27));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        pushButton = new QPushButton(layoutWidget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setFont(font);

        horizontalLayout->addWidget(pushButton);


        retranslateUi(exitDialog);

        QMetaObject::connectSlotsByName(exitDialog);
    } // setupUi

    void retranslateUi(QDialog *exitDialog)
    {
        exitDialog->setWindowTitle(QCoreApplication::translate("exitDialog", "Dialog", nullptr));
        label->setText(QCoreApplication::translate("exitDialog", "  \347\241\256\345\256\232\346\226\255\345\274\200\351\223\276\346\216\245\357\274\237", nullptr));
        pushButton->setText(QCoreApplication::translate("exitDialog", "certain", nullptr));
    } // retranslateUi

};

namespace Ui {
    class exitDialog: public Ui_exitDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_EXITDIALOG_H
