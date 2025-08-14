/********************************************************************************
** Form generated from reading UI file 'detailsscreen.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DETAILSSCREEN_H
#define UI_DETAILSSCREEN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_detailsscreen
{
public:
    QToolButton *closebutton;
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;

    void setupUi(QDialog *detailsscreen)
    {
        if (detailsscreen->objectName().isEmpty())
            detailsscreen->setObjectName(QString::fromUtf8("detailsscreen"));
        detailsscreen->resize(320, 240);
        detailsscreen->setStyleSheet(QString::fromUtf8("#detailsscreen {\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));\n"
"}\n"
""));
        closebutton = new QToolButton(detailsscreen);
        closebutton->setObjectName(QString::fromUtf8("closebutton"));
        closebutton->setGeometry(QRect(265, 10, 41, 16));
        closebutton->setStyleSheet(QString::fromUtf8("#closebutton {\n"
"    background-color: transparent;  /* \351\200\217\346\230\216\350\203\214\346\231\257 */\n"
"    border: none;                     /* \346\227\240\350\276\271\346\241\206 */\n"
"}"));
        widget = new QWidget(detailsscreen);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setGeometry(QRect(20, 30, 261, 141));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(widget);
        label->setObjectName(QString::fromUtf8("label"));
        QFont font;
        font.setItalic(true);
        font.setUnderline(true);
        label->setFont(font);
        label->setStyleSheet(QString::fromUtf8("color: rgb(200, 200, 200); /* \346\267\261\347\201\260\350\211\262\346\226\207\345\255\227 */"));

        verticalLayout->addWidget(label);

        label_2 = new QLabel(widget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFont(font);
        label_2->setStyleSheet(QString::fromUtf8("color: rgb(200, 200, 200); /* \346\267\261\347\201\260\350\211\262\346\226\207\345\255\227 */"));

        verticalLayout->addWidget(label_2);

        label_3 = new QLabel(widget);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setFont(font);
        label_3->setStyleSheet(QString::fromUtf8("color: rgb(200, 200, 200); /* \346\267\261\347\201\260\350\211\262\346\226\207\345\255\227 */"));

        verticalLayout->addWidget(label_3);


        retranslateUi(detailsscreen);

        QMetaObject::connectSlotsByName(detailsscreen);
    } // setupUi

    void retranslateUi(QDialog *detailsscreen)
    {
        detailsscreen->setWindowTitle(QCoreApplication::translate("detailsscreen", "Dialog", nullptr));
        closebutton->setText(QCoreApplication::translate("detailsscreen", "close", nullptr));
        label->setText(QCoreApplication::translate("detailsscreen", "RMBG-1.4-1.3b\345\210\206\345\211\262\346\250\241\345\236\213", nullptr));
        label_2->setText(QCoreApplication::translate("detailsscreen", "\344\270\212\344\270\213\346\226\207\346\216\250\347\220\206\346\225\260\351\207\217<=4", nullptr));
        label_3->setText(QCoreApplication::translate("detailsscreen", "\346\234\200\345\244\247\346\211\271\351\207\217\350\256\276\345\256\232<=4", nullptr));
    } // retranslateUi

};

namespace Ui {
    class detailsscreen: public Ui_detailsscreen {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DETAILSSCREEN_H
