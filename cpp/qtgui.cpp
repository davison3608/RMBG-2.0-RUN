#include "qtpacked.h"

//! \brief branchscreen对象
branchscreen::branchscreen(QWidget *parent): 
QDialog(parent), detailsui(new Ui::detailsscreen)
{
    qDebug()<<"构造 detailsui成员";
    qDebug()<<"setup detailsui成员";
    this->detailsui->setupUi(this);

    qDebug()<<"已准备分支界面details";
}
branchscreen::~branchscreen()
{
    delete this->detailsui;
}

void branchscreen::on_closebutton_clicked() noexcept
{
    qDebug()<<"\n\n触发事件 关闭分支界面details\n\n";
    //关闭分支界面 隐藏窗口并非析构branchscreen对象
    this->close();
    qDebug()<<"已回到初始化界面";
}

//! \brief SplashScreen对象
SplashScreen::SplashScreen(QWidget *parent): 
QWidget(parent), ui(new Ui::Form)
{
    qDebug()<<"构造 ui成员";
    qDebug()<<"setup ui成员";
    this->ui->setupUi(this);

    qDebug()<<"已准备初始化界面";
}
SplashScreen::~SplashScreen()
{
    //析构this->details对象
    delete this->ui;
}

void SplashScreen::on_informationbutton_clicked() noexcept
{
    qDebug()<<"\n\n触发事件 弹出分支界面details\n\n";
    //构造分支界面对象
    this->detailui=new branchscreen();
    //分支界面显示
    this->detailui->show();
}

void SplashScreen::on_toolButton_2_clicked()
{
    qDebug()<<"\n\n触发事件 准备确认创建对话窗口\n\n";

    qDebug()<<"确认line edit输入";
    QString numcontexts=this->ui->lineEdit->text();
    QString numbatchsizes=this->ui->lineEdit_2->text();
    
    if (numcontexts.isEmpty())
        qDebug()<<"contexts num is empty !";
    else
        qDebug()<<"contexts num = "<<numcontexts;

    if (numbatchsizes.isEmpty())
        qDebug()<<"numbatchsizes num is empty !";
    else
        qDebug()<<"numbatchsizes num = "<<numbatchsizes;

    //尝试qstring内容为数字
    bool cflag=false;
    bool bflag=false;
    uint cnum=4;
    uint bnum=1;
    //uint cnum=numcontexts.toUInt(&cflag);
    //uint bnum=numbatchsizes.toUInt(&bflag);
    if (cflag || bflag) {
        qDebug()<<"输入context和batchsize不为数字 !";
        return ;
    }    

    //确认设定后构造对话界面对象
    if ((cnum <= 0) || (bnum <= 0)) {
        qDebug()<<"输入context和batchsize设定不符合要求 !";
        return ;
    }
    else if ((cnum > 4) || (bnum > 1)) {
        qDebug()<<"输入context和batchsize设定不符合要求 !";
        return ;
    }

    //构造talkscreen对象
    this->talkui=new talkscreen(cnum, bnum, nullptr);
    //对话界面显示
    this->talkui->show();    

    //关闭当前初始界面
    this->close();
}

