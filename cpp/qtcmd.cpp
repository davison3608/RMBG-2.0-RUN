#include "qtpacked.h"

//
//! cmdscreen对象
cmdscreen::cmdscreen(QWidget *parent): 
QDialog(parent), cmddialogui(new Ui::cmdDialog)
{ this->cmddialogui->setupUi(this); }

cmdscreen::~cmdscreen()
{ delete this->cmddialogui; }

void cmdscreen::on_closeButton_clicked()
{ 
    _qtexts s=this->cmddialogui->cmdEdit->text();

    //获取lineedit
    this->qs_cmd=s;
    this->cs_cmd=s.toStdString();

    this->close(); 
}

//
//! talkscreen对象
bool blockcmdsend_single(int& _sendcmdsocket, const char *buffer, TypeAlias::_usize allsize)
{
    //已发送的数据字节
    size_t sendsize=0;

    while (sendsize < allsize) {
        ssize_t cn=::send(
            _sendcmdsocket,
            buffer + sendsize,
            allsize - sendsize,
            0
        );
        
        if (cn == 0) {
            qDebug()<<"发送期间服务器断开连接 发送cmd失败 !";
            return false;
        }
        else if (cn < 0) {
            if (errno == ACCEPTNONE) {
                qDebug()<<"发送缓冲区满 发送cmd失败 !";        
                return false;
            }
            qDebug()<<"发送错误 连接状态有误 发送cmd失败 !";
            return false;
        }
        sendsize+=cn;
    }
    return true;    
}

ServerCmd CStringToServerCmd(std::string s) noexcept
{
    //迭代器指向
    auto it=_Map_ServerCmd.find(s);
    //未找到映射 无效命令
    if (it == _Map_ServerCmd.end()) 
        return ServerCmd::INVALID;
    //无效命令        
    if (it->second == ServerCmd::INVALID)
        return ServerCmd::INVALID;

    else if (it->second == ServerCmd::RECEIVEDOWN)
        return ServerCmd::RECEIVEDOWN;
    
    else if (it->second == ServerCmd::RECEIVEON)
        return ServerCmd::RECEIVEON;

    else if (it->second == ServerCmd::STATUS)
        return ServerCmd::STATUS;

    else if (it->second == ServerCmd::TEST)
        return ServerCmd::TEST;

    else if (it->second == ServerCmd::TRTSET)
        return ServerCmd::TRTSET;
}

std::string ServerCmdToCString(ServerCmd s) noexcept
{
    //迭代器指向
    auto it=_Map_ServerString.find(s);
    //未找到映射 无效命令
    if (it == _Map_ServerString.end()) 
        return "invalid";
    //无效命令        
    if (it->second == "invalid")
        return "invalid";

    else if (it->second == "status")
        return "status";
    
    else if (it->second == "receivedown")
        return "receivedown";

    else if (it->second == "receiveon")
        return "receiveon";

    else if (it->second == "test")
        return "test";

    else if (it->second == "trtset")
        return "trtset";
}

void talkscreen::on_cmdButton_clicked()
{
    qDebug()<<"\n\n触发事件 进入cmd界面\n\n";
    //构造cmdscreen
    this->cmddialogui=new cmdscreen();
    //阻塞显示
    this->cmddialogui->exec();

    //关闭窗口
    //获取字符串
    _ctexts cmd_str=this->cmddialogui->cs_cmd;

    if (cmd_str.empty()) {
        qDebug()<<"没有cmd输入 已返回";
        return ;
    }

    //解析字符串到对应枚举
    ServerCmd cmd;
    cmd=CStringToServerCmd(cmd_str);
    //发送指令
    std::cout<<"发送指令 "<<cmd_str<<"\n";
    if (!this->blockcmdsend(cmd, this->cus_cmd_socket)) 
        return ;

    std::cout<<"成功发送cmd命令 " + cmd_str<<"\n";
    return ;
}

bool talkscreen::blockcmdsend(const ServerCmd _ser_cmd, int& _sendsocket)
{
    bool status;
    
    //枚举变量为整数类型
    int cmd_int=static_cast<int>(_ser_cmd);
    const char *cmd_int_p=reinterpret_cast<const char*>(&cmd_int);

    status=blockcmdsend_single(
        this->cus_cmd_socket,
        cmd_int_p,
        sizeof(cmd_int)
    );

    //对于trtset命令额外发送
    if (_ser_cmd == ServerCmd::TRTSET) {
        uint numset[2]={this->cnum, this->bnum};
        const char *numset_p=reinterpret_cast<char*>(numset);
        status=blockcmdsend_single(
            this->cus_cmd_socket,
            numset_p,
            2 * sizeof(uint)
        );
    }

    return status;
}
