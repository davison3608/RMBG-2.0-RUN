/********************************************************************************
** Form generated from reading UI file 'cmddialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CMDDIALOG_H
#define UI_CMDDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_cmdDialog
{
public:
    QHBoxLayout *horizontalLayout;
    QLineEdit *cmdEdit;
    QPushButton *closeButton;

    void setupUi(QDialog *cmdDialog)
    {
        if (cmdDialog->objectName().isEmpty())
            cmdDialog->setObjectName(QString::fromUtf8("cmdDialog"));
        cmdDialog->resize(322, 44);
        horizontalLayout = new QHBoxLayout(cmdDialog);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        cmdEdit = new QLineEdit(cmdDialog);
        cmdEdit->setObjectName(QString::fromUtf8("cmdEdit"));

        horizontalLayout->addWidget(cmdEdit);

        closeButton = new QPushButton(cmdDialog);
        closeButton->setObjectName(QString::fromUtf8("closeButton"));

        horizontalLayout->addWidget(closeButton);


        retranslateUi(cmdDialog);

        QMetaObject::connectSlotsByName(cmdDialog);
    } // setupUi

    void retranslateUi(QDialog *cmdDialog)
    {
        cmdDialog->setWindowTitle(QApplication::translate("cmdDialog", "Dialog", nullptr));
        cmdEdit->setText(QApplication::translate("cmdDialog", "\350\276\223\345\205\245\346\214\207\344\273\244", nullptr));
        closeButton->setText(QApplication::translate("cmdDialog", "savecmd", nullptr));
    } // retranslateUi

};

namespace Ui {
    class cmdDialog: public Ui_cmdDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CMDDIALOG_H
