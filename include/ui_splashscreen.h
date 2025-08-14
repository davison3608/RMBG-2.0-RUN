/********************************************************************************
** Form generated from reading UI file 'splashscreen.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SPLASHSCREEN_H
#define UI_SPLASHSCREEN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Form
{
public:
    QVBoxLayout *verticalLayout_2;
    QWidget *topwidget;
    QVBoxLayout *verticalLayout_3;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QPushButton *informationbutton;
    QSpacerItem *horizontalSpacer_5;
    QSpacerItem *verticalSpacer_2;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_2;
    QLabel *versionlabel;
    QSpacerItem *verticalSpacer;
    QWidget *widget_3;
    QHBoxLayout *horizontalLayout_4;
    QSpacerItem *horizontalSpacer_6;
    QFrame *frame_4;
    QHBoxLayout *horizontalLayout_5;
    QToolButton *toolButton_2;
    QSpacerItem *horizontalSpacer_4;
    QVBoxLayout *verticalLayout;
    QLineEdit *lineEdit;
    QLineEdit *lineEdit_2;
    QSpacerItem *horizontalSpacer_7;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;

    void setupUi(QWidget *Form)
    {
        if (Form->objectName().isEmpty())
            Form->setObjectName(QString::fromUtf8("Form"));
        Form->resize(479, 615);
        Form->setStyleSheet(QString::fromUtf8("QWidget {\n"
"	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));\n"
"}"));
        verticalLayout_2 = new QVBoxLayout(Form);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        topwidget = new QWidget(Form);
        topwidget->setObjectName(QString::fromUtf8("topwidget"));
        topwidget->setLayoutDirection(Qt::RightToLeft);
        topwidget->setStyleSheet(QString::fromUtf8("* {\n"
"    background: none;\n"
"    border: none;\n"
"    color: palette(text);\n"
"}"));
        verticalLayout_3 = new QVBoxLayout(topwidget);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        widget = new QWidget(topwidget);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setMaximumSize(QSize(16777215, 136));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        informationbutton = new QPushButton(widget);
        informationbutton->setObjectName(QString::fromUtf8("informationbutton"));

        horizontalLayout->addWidget(informationbutton);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_5);


        verticalLayout_3->addWidget(widget);

        verticalSpacer_2 = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Minimum);

        verticalLayout_3->addItem(verticalSpacer_2);

        widget1 = new QWidget(topwidget);
        widget1->setObjectName(QString::fromUtf8("widget1"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(widget1->sizePolicy().hasHeightForWidth());
        widget1->setSizePolicy(sizePolicy);
        horizontalLayout_2 = new QHBoxLayout(widget1);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        versionlabel = new QLabel(widget1);
        versionlabel->setObjectName(QString::fromUtf8("versionlabel"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(versionlabel->sizePolicy().hasHeightForWidth());
        versionlabel->setSizePolicy(sizePolicy1);
        versionlabel->setMinimumSize(QSize(400, 400));
        versionlabel->setStyleSheet(QString::fromUtf8("QLabel {\n"
"\n"
"}"));
        versionlabel->setPixmap(QPixmap(QString::fromUtf8("../file/9357-CuXJvW.jpg")));
        versionlabel->setScaledContents(true);
        versionlabel->setWordWrap(true);

        horizontalLayout_2->addWidget(versionlabel);


        verticalLayout_3->addWidget(widget1);

        verticalSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Minimum);

        verticalLayout_3->addItem(verticalSpacer);

        widget_3 = new QWidget(topwidget);
        widget_3->setObjectName(QString::fromUtf8("widget_3"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(widget_3->sizePolicy().hasHeightForWidth());
        widget_3->setSizePolicy(sizePolicy2);
        horizontalLayout_4 = new QHBoxLayout(widget_3);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_6);

        frame_4 = new QFrame(widget_3);
        frame_4->setObjectName(QString::fromUtf8("frame_4"));
        horizontalLayout_5 = new QHBoxLayout(frame_4);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        toolButton_2 = new QToolButton(frame_4);
        toolButton_2->setObjectName(QString::fromUtf8("toolButton_2"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(toolButton_2->sizePolicy().hasHeightForWidth());
        toolButton_2->setSizePolicy(sizePolicy3);
        toolButton_2->setMinimumSize(QSize(60, 0));

        horizontalLayout_5->addWidget(toolButton_2);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_4);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        lineEdit = new QLineEdit(frame_4);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        lineEdit->setMinimumSize(QSize(40, 0));
        QFont font;
        font.setBold(true);
        font.setItalic(false);
        font.setWeight(75);
        lineEdit->setFont(font);

        verticalLayout->addWidget(lineEdit);

        lineEdit_2 = new QLineEdit(frame_4);
        lineEdit_2->setObjectName(QString::fromUtf8("lineEdit_2"));
        lineEdit_2->setFont(font);
        lineEdit_2->setCursorMoveStyle(Qt::LogicalMoveStyle);

        verticalLayout->addWidget(lineEdit_2);


        horizontalLayout_5->addLayout(verticalLayout);


        horizontalLayout_4->addWidget(frame_4);

        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_7);


        verticalLayout_3->addWidget(widget_3);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label = new QLabel(topwidget);
        label->setObjectName(QString::fromUtf8("label"));
        QFont font1;
        font1.setFamily(QString::fromUtf8("Ubuntu Mono"));
        font1.setItalic(true);
        label->setFont(font1);
        label->setStyleSheet(QString::fromUtf8(""));

        horizontalLayout_3->addWidget(label);


        verticalLayout_3->addLayout(horizontalLayout_3);


        verticalLayout_2->addWidget(topwidget);


        retranslateUi(Form);

        QMetaObject::connectSlotsByName(Form);
    } // setupUi

    void retranslateUi(QWidget *Form)
    {
        Form->setWindowTitle(QApplication::translate("Form", "Form", nullptr));
        informationbutton->setText(QApplication::translate("Form", "information", nullptr));
        versionlabel->setText(QString());
        toolButton_2->setText(QApplication::translate("Form", "create", nullptr));
        lineEdit->setText(QString());
        lineEdit->setPlaceholderText(QApplication::translate("Form", "numbercontexts=4", nullptr));
        lineEdit_2->setText(QString());
        lineEdit_2->setPlaceholderText(QApplication::translate("Form", "maxbatchsizes=1", nullptr));
#ifndef QT_NO_TOOLTIP
        label->setToolTip(QApplication::translate("Form", "<html><head/><body><p align=\"justify\">\345\210\235\345\247\213\347\225\214\351\235\242\346\214\211\350\246\201\346\261\202\345\241\253\345\205\245\344\277\241\346\201\257</p></body></html>", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_WHATSTHIS
        label->setWhatsThis(QApplication::translate("Form", "<html><head/><body><p align=\"justify\"><br/></p></body></html>", nullptr));
#endif // QT_NO_WHATSTHIS
        label->setText(QApplication::translate("Form", "             \345\210\235\345\247\213\347\225\214\351\235\242\350\216\267\345\217\226context\344\270\216batchsize\344\277\241\346\201\257", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Form: public Ui_Form {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SPLASHSCREEN_H
