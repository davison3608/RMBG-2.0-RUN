/********************************************************************************
** Form generated from reading UI file 'talkformscreen.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TALKFORMSCREEN_H
#define UI_TALKFORMSCREEN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TalkForm
{
public:
    QHBoxLayout *horizontalLayout_2;
    QWidget *leftwidget;
    QVBoxLayout *verticalLayout;
    QPushButton *contectbutton;
    QSpacerItem *verticalSpacer_3;
    QPushButton *sendbutton;
    QSpacerItem *verticalSpacer;
    QPushButton *exitbutton;
    QWidget *talkwidget;
    QLineEdit *iplineedit;
    QLineEdit *picturepathlineedit;
    QLineEdit *portlineedit;
    QLineEdit *namelineedit;
    QLineEdit *cmdportlineedit;
    QPushButton *cmdButton;

    void setupUi(QWidget *TalkForm)
    {
        if (TalkForm->objectName().isEmpty())
            TalkForm->setObjectName(QString::fromUtf8("TalkForm"));
        TalkForm->resize(653, 157);
        horizontalLayout_2 = new QHBoxLayout(TalkForm);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        leftwidget = new QWidget(TalkForm);
        leftwidget->setObjectName(QString::fromUtf8("leftwidget"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(3);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(leftwidget->sizePolicy().hasHeightForWidth());
        leftwidget->setSizePolicy(sizePolicy);
        leftwidget->setStyleSheet(QString::fromUtf8("QWidget#leftwidget {\n"
"	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(111, 111, 111, 255), stop:1 rgba(255, 255, 255, 255));\n"
"}"));
        verticalLayout = new QVBoxLayout(leftwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        contectbutton = new QPushButton(leftwidget);
        contectbutton->setObjectName(QString::fromUtf8("contectbutton"));
        QFont font;
        font.setItalic(true);
        contectbutton->setFont(font);
        contectbutton->setStyleSheet(QString::fromUtf8("#contectbutton { /* \346\263\250\346\204\217#\345\217\267 */\n"
"   	background-color: rgb(32, 74, 135);\n"
"}"));

        verticalLayout->addWidget(contectbutton);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_3);

        sendbutton = new QPushButton(leftwidget);
        sendbutton->setObjectName(QString::fromUtf8("sendbutton"));
        sendbutton->setFont(font);

        verticalLayout->addWidget(sendbutton);

        verticalSpacer = new QSpacerItem(20, 264, QSizePolicy::Minimum, QSizePolicy::Preferred);

        verticalLayout->addItem(verticalSpacer);

        exitbutton = new QPushButton(leftwidget);
        exitbutton->setObjectName(QString::fromUtf8("exitbutton"));
        exitbutton->setStyleSheet(QString::fromUtf8("#exitbutton { /* \346\263\250\346\204\217#\345\217\267 */   	\n"
"	background-color: rgb(164, 0, 0);\n"
"}"));

        verticalLayout->addWidget(exitbutton);


        horizontalLayout_2->addWidget(leftwidget);

        talkwidget = new QWidget(TalkForm);
        talkwidget->setObjectName(QString::fromUtf8("talkwidget"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(7);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(talkwidget->sizePolicy().hasHeightForWidth());
        talkwidget->setSizePolicy(sizePolicy1);
        talkwidget->setStyleSheet(QString::fromUtf8("QWidget#talkwidget {\n"
"	background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0.0340909, stop:0 rgba(111, 111, 111, 255), stop:1 rgba(255, 255, 255, 255));\n"
"}"));
        iplineedit = new QLineEdit(talkwidget);
        iplineedit->setObjectName(QString::fromUtf8("iplineedit"));
        iplineedit->setGeometry(QRect(20, 20, 201, 31));
        picturepathlineedit = new QLineEdit(talkwidget);
        picturepathlineedit->setObjectName(QString::fromUtf8("picturepathlineedit"));
        picturepathlineedit->setGeometry(QRect(20, 60, 201, 31));
        portlineedit = new QLineEdit(talkwidget);
        portlineedit->setObjectName(QString::fromUtf8("portlineedit"));
        portlineedit->setGeometry(QRect(230, 20, 81, 31));
        namelineedit = new QLineEdit(talkwidget);
        namelineedit->setObjectName(QString::fromUtf8("namelineedit"));
        namelineedit->setGeometry(QRect(230, 60, 181, 31));
        cmdportlineedit = new QLineEdit(talkwidget);
        cmdportlineedit->setObjectName(QString::fromUtf8("cmdportlineedit"));
        cmdportlineedit->setGeometry(QRect(320, 20, 91, 31));
        cmdButton = new QPushButton(talkwidget);
        cmdButton->setObjectName(QString::fromUtf8("cmdButton"));
        cmdButton->setGeometry(QRect(20, 100, 41, 21));

        horizontalLayout_2->addWidget(talkwidget);


        retranslateUi(TalkForm);

        QMetaObject::connectSlotsByName(TalkForm);
    } // setupUi

    void retranslateUi(QWidget *TalkForm)
    {
        TalkForm->setWindowTitle(QApplication::translate("TalkForm", "Form", nullptr));
        contectbutton->setText(QApplication::translate("TalkForm", "contect", nullptr));
        sendbutton->setText(QApplication::translate("TalkForm", "send", nullptr));
        exitbutton->setText(QApplication::translate("TalkForm", "exit", nullptr));
        iplineedit->setPlaceholderText(QApplication::translate("TalkForm", "\350\276\223\345\205\245\345\276\205\351\223\276\346\216\245\347\232\204\346\234\215\345\212\241\347\253\257ip\345\234\260\345\235\200", nullptr));
        picturepathlineedit->setPlaceholderText(QApplication::translate("TalkForm", "\350\276\223\345\205\245\345\276\205\345\244\204\347\220\206\347\232\204\345\233\276\345\203\217\347\273\235\345\257\271\350\267\257\345\276\204", nullptr));
        portlineedit->setPlaceholderText(QApplication::translate("TalkForm", "\351\200\232\344\277\241\347\253\257\345\217\243", nullptr));
        namelineedit->setText(QString());
        namelineedit->setPlaceholderText(QApplication::translate("TalkForm", "\344\277\235\345\255\230\345\233\276\345\203\217\346\226\207\344\273\266\345\220\215", nullptr));
        cmdportlineedit->setPlaceholderText(QApplication::translate("TalkForm", "cmd\347\253\257\345\217\243", nullptr));
        cmdButton->setText(QApplication::translate("TalkForm", "cmd", nullptr));
    } // retranslateUi

};

namespace Ui {
    class TalkForm: public Ui_TalkForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TALKFORMSCREEN_H
