#include "ui_exitdialog.h" //退出弹窗界面
#include "ui_talkformscreen.h" //对话界面
#include "ui_cmddialog.h" 
#include "ui_splashscreen.h" //初始化ui界面
#include "ui_detailsscreen.h" //弹出界面

#include <QStringListModel> 
#include<QDebug>

#ifndef SockDatas
#define SockDatas
//! \brief 默认服务端监听端口 管理端口
#define SERLISPORT  8080
#define SERCMDPORT  8888

#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <cstdint>

//! \brief ipv4协议族 流数据方式
#define IPV4  AF_INET
#define STREAMSR  SOCK_STREAM

//! \brief socket标志设置
#define SOCKFLAGGET F_GETFL
#define SOCKFLAGSET F_SETFL

#define SOCKISBLOCK O_NONBLOCK
#define SOCKFLAGBLOCK(x)  (x & ~SOCKISBLOCK)
#define SOCKFLAGNOBLOCK(x)  (x | SOCKISBLOCK)

//! \brief accept recv非阻塞读空返回状态
#define ACCEPTNONE  EAGAIN

/**
 * \brief using类   
*/
class TypeAlias
{
public:
    TypeAlias() = default;

    virtual ~TypeAlias() = default;

    using _usize = std::size_t;
    using _size = ssize_t;
    using _qtuint = quint32;
    using _netflag = std::atomic<bool>; 
    using _filecheck = std::ifstream;

    using _ctexts = std::string;
    using _qtexts = QString;
    
    using _pro_mtx = pthread_mutex_t;
    using _flag = std::atomic<bool>;
    using _status = __int8_t;
};

/**
 * \brief tcp传递的数据流结构体
*/
struct SQMat 
{
    //! \brief 图像绝对路径字节长度 图像绝对路径
    //! \brief 接收时不指定
    TypeAlias::_usize matpath_len;
    TypeAlias::_ctexts matpath;

    //! \brief 图像名称字节长度 图像名称
    TypeAlias::_usize matname_len;
    TypeAlias::_ctexts matname;
 
    //! \brief 图像字节长度 图像数据指针 
    //! \brief 动态分配后由外部reset
    TypeAlias::_usize mat_len;
    char *matdatas=nullptr;

    //! \brief 提前计算动态structsize大小 
    const SQMat *init(TypeAlias::_ctexts& path, TypeAlias::_ctexts& name);

    //! \brief 深度拷贝
    void deepcopy(SQMat& m);

    //! \brief 重置处理所有成员
    void reset();

    //! \brief 总体字节大小
    TypeAlias::_usize totall_len; 

    //! \brief 测试使用 仅reset生效
    TypeAlias::_ctexts test_str;
};

#include <unistd.h> //fork getpid 
#include <sys/wait.h> //waitpid
#include <cstdlib>    //exit
#include <sys/sem.h>  //信号量

//! \brief 信号量定义
#define SEMGET_CREATE  0666 | IPC_CREAT
#define SEMCTL_SET  SETVAL
#define SEMCTL_REMOVE  IPC_RMID

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
//! \brief 共享内存创建与删除
#define SHARECREATE  IPC_CREAT | 0666
#define SHAREGET  0666
#define SHAREREMOVE  IPC_RMID

/**
 * \brief 进程共享对象 
*/
class ProcessShare: virtual public TypeAlias
{
public:
    using TypeAlias::_pro_mtx;
    using TypeAlias::_flag;
    using TypeAlias::_status;
    
    ProcessShare() = default;

    ~ProcessShare() = default;

    //！手动构造
    static ProcessShare *init();
    //! 手动获取对象指针 手动销毁或减少引用
    static ProcessShare *get();
    void destroy();

    //! 将对象存入缓冲区
    _status push(SQMat& _push);

    //! 将对象取出缓冲区
    _status pop(SQMat& _pop);

    using TypeAlias::_usize;
    //! 缓冲有效
    _flag iscache{false};

private:
    //! 共享内存标识
    int share_id;

    //! 引用计数
    _usize ref_count=0;

    //! 缓冲更改锁
    _pro_mtx send_mt;
    //! 缓冲结构体指针
    SQMat cache;
    //! 缓冲有效通知信号量
    int sem_cache;
};

//! \brief 声明计算对象
class Qdeque;

template<int NUMTHREADS>
class rtts;

#define THREADS  6
#define CONTEXTS  4

//! \brief 结构体数据传递状态指示
enum class SQMatStatus
{
    TOTALL_LEN,     //总体字节长度
    MATPATH_LEN,    //路径长度
    MATNAME_LEN,    //名称长度
    MAT_LEN,        //图像数据长度
    MATPATH,        //路径内容
    MATNAME,        //名称内容
    MATDATAS,       //图像数据
};

/**
 * \brief 命令枚举类
*/
enum class ServerCmd
{
    INVALID ,       //未知命令
    STATUS ,        //检查服务器状态
    RECEIVEDOWN ,   //停止推理 丢失正在发送数据与接收数据
    RECEIVEON ,     //继续推理
    TEST ,          //测试命令
    TRTSET          //推理设定命令
};

/**
 * \brief 设定字符串到命令映射表
*/
const std::map<TypeAlias::_ctexts, ServerCmd> _Map_ServerCmd{
    {"invalid", ServerCmd::INVALID}, 
    {"status", ServerCmd::STATUS}, 
    {"receivedown", ServerCmd::RECEIVEDOWN}, 
    {"receiveon", ServerCmd::RECEIVEON},
    {"test", ServerCmd::TEST}, 
    {"trtset", ServerCmd::TRTSET}
};
const std::map<ServerCmd, TypeAlias::_ctexts> _Map_ServerString{
    {ServerCmd::INVALID, "invalid"}, 
    {ServerCmd::STATUS, "status"}, 
    {ServerCmd::RECEIVEDOWN, "receivedown"}, 
    {ServerCmd::RECEIVEON, "receiveon"},
    {ServerCmd::TEST, "test"}, 
    {ServerCmd::TRTSET, "trtset"}
};

//! \brief 映射函数
//! \brief 字符串定位到枚举
ServerCmd CStringToServerCmd(std::string s) noexcept;
std::string ServerCmdToCString(ServerCmd s) noexcept;

#endif

#ifndef TALKSCREEN
#define TALKSCREEN
//网络申请
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/**
 * \brief 退出弹窗界面对象
*/
class exitscreen: public QDialog
{
    Q_OBJECT

public:
    explicit exitscreen(bool& checked, QWidget *parent=nullptr);
    
    ~exitscreen();

private slots:
    //! 触发确认退出
    void on_pushButton_clicked();

private:
    //! exitdialog对象指针
    Ui::exitDialog *exitdialogui;

public:
    bool checked;
};

/**
 * \brief cmd弹窗界面对象
*/
class cmdscreen: public QDialog, public TypeAlias
{
    Q_OBJECT

public:
    using TypeAlias::_ctexts;
    using TypeAlias::_qtexts;

    explicit cmdscreen(QWidget *parent=nullptr);
    
    ~cmdscreen();

private slots:
    //! 触发确认退出 保存输入
    void on_closeButton_clicked();  

private:
    //! exitdialog对象指针
    Ui::cmdDialog *cmddialogui;

public:
    //! 输入命令字符串 外部可访问
    _qtexts qs_cmd;
    _ctexts cs_cmd;
};

//! \brief 声明接收结构体容器
extern std::vector<SQMat> sqmat_vec;
//! \brief 声明接收索引
extern size_t sqmat_id;

//! \brief 声明通过标志位表明接收的是枚举还是拆分数据
extern SQMatStatus sqmat_tus;
extern bool isenum;

/**
 * \brief talkscreen对话界面对象
*/
class talkscreen: public QWidget, public TypeAlias
{
    Q_OBJECT

public:
    using TypeAlias::_usize;
    using TypeAlias::_size;
    using TypeAlias::_qtuint;
    using TypeAlias::_netflag; 
    using TypeAlias::_filecheck;
    using TypeAlias::_ctexts;
    using TypeAlias::_qtexts;
    using _runflag = TypeAlias::_netflag;

    talkscreen() = delete;
    //! 传递上下文与批量信息
    explicit talkscreen(_qtuint& numcontexts, _qtuint& numbatchsize, QWidget *parent=nullptr);

    ~talkscreen();

    //! 来自SplashScreen的lineedit设定
    uint cnum;
    uint bnum;

private slots:
    //! \brief 点击退出按钮触发事件 退出程序或是断开连接同步后退出
    void on_exitbutton_clicked();

    //! \brief 点击链接按钮触发事件 根据ip建立链接
    void on_contectbutton_clicked();

    //! \brief 点击触发事件 已连接状态下发送图片数据
    //! \brief 在推理线程池中列队
    void on_sendbutton_clicked();

    //! \brief 点击触发事件 已连接状态下发出暂停推理申请
    //! \brief 同步到当前正在推理图片结束
    void on_downbutton_clicked();

    //! \brief 点击触发事件 阻塞开启cmd界面 解析字符串为指令
    void on_cmdButton_clicked();

private:
    //! \brief 连接服务端
    void serverconnected();

    //! \brief 连接服务端管理端口
    void servercmdconnected();

    //! \brief 关闭服务端连接
    void serverclose();

private:
    //! \brief 异步接收 线程开启
    void asyncreceive();

    //! \brief 主动发送数据
    //! \brief 阻塞式循环发送一次结构数据
    bool blocksend(const SQMat *&_sqmat, int& _sendsocket);

    //! \brief 主动发送命令
    //! \brief 连接通过后设定命令发送一次 传递设定的推理信息
    bool blockcmdsend(const ServerCmd _ser_cmd, int& _sendsocket);    

private:
    //! talkform对象指针
    Ui::TalkForm *talkformui;
    //! exitscreen对象指针
    exitscreen *exitdialogui;
    //! cmd界面对象指针
    cmdscreen *cmddialogui;

private:
    //! 链接状态指示
    _netflag iscontected;
    //! 推理暂停状态指示
    //_runflag isrun;

    //! 客户端通信socket 管理socket
    int cus_socket;
    int cus_cmd_socket;

    //! 服务端ip地址
    _ctexts ser_ip;
    //! 服务端端口号 cmd端口号
    _qtuint ser_port;
    _qtuint ser_cmd_port;

private:
    //! 计算进程共享对象
    ProcessShare *_share;
    //! 计算线程池对象
    rtts<THREADS> *inference;
};

#endif

#ifndef DETAILSSCREEN
#define DETAILSSCREEN
/**
 * \brief 分支界面对象
*/
class branchscreen : public QDialog
{
    Q_OBJECT

public:
    //! \brief 同时new details screen成员对象
    explicit branchscreen(QWidget *parent=nullptr);

    ~branchscreen();

private slots:
    //! \brief 按钮点击事件触发 close隐藏当前窗口
    void on_closebutton_clicked() noexcept;

private:
    //! details界面对象指针
    Ui::detailsscreen *detailsui;
};

#endif

#ifndef SPLASHSCREEN
#define SPLASHSCREEN
/**
 * \brief 初始化界面对象
*/
class SplashScreen : public QWidget {
    Q_OBJECT

public:
    //! \brief 基类指针 同时new form成员对象
    explicit SplashScreen(QWidget *parent=nullptr);
    
    ~SplashScreen();

private slots:
    //! \brief 按钮点击事件触发 说明界面
    void on_informationbutton_clicked() noexcept;

    //! \brief 按钮触发 创建对话窗口
    void on_toolButton_2_clicked();

private:
    //! \brief form界面对象指针
    //! \param lineEdit对象输入槽 上下文数量
    //! \param lineEdit_2对象输入槽 批量数量
    Ui::Form *ui;

    //! talkformui弹出界面对象
    talkscreen *talkui;

    //! details弹出界面对象
    branchscreen *detailui;
};

#endif

