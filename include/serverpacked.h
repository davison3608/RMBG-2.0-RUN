#ifndef Server
#define Server
#include "qtpacked.h"

class baseServer: public TypeAlias
{
public:
    using TypeAlias::_ctexts;
    using TypeAlias::_qtexts;
    using TypeAlias::_netflag;
    using TypeAlias::_size;
    using TypeAlias::_usize;
    using _port = TypeAlias::_qtuint;
    using _sock = int;

    baseServer() noexcept = default;

    virtual ~baseServer() noexcept = default; 

protected:
    virtual void setsocketdefault() {}

    virtual void setepolleventdefault() {}

    _ctexts _ser_ip;

    //! \brief 维护监听和管理端口
    _port _ser_port;
    _port _ser_cmd_port;
};

#include <mutex>
#include <condition_variable>

//! \brief socket监听队列长度
#define LISSOCCOM  SOMAXCONN
#define LISSOCOPT  LISSOCCOM/2
#define LISSOCMIN  LISSOCOPT/2  

//! \brief epoll默认处理事件长度
#define DEFAULTEVENTS  10

//! \brief epoll add/del
#define EPOLLADD  EPOLL_CTL_ADD
#define EPOLLDEL  EPOLL_CTL_DEL
#define EPOLLMOD  EPOLL_CTL_MOD

//! \brief epoll event type
#define EVENTRECV  EPOLLIN
#define EVENTSEND  EPOLLOUT
#define EVENTET  EPOLLET //et
#define EVENTLT  EPOLLIN //lt

#define EVENTCLOSE  EPOLLHUP //socket关闭事件标志
#define EVENTERR  EPOLLERR //socket错误 关闭伴随触发

//! \brief 非阻塞下send recv返回状态
#define SENDCLOSE(x)  ((x == 0)) //是否发送时对方断开
#define SENDNONE  EWOULDBLOCK //已发送完数据
#define RECVCLOSE(x)  ((x == 0)) //是否接收时对断开
#define RECVNONE  EWOULDBLOCK //已接收完数据

//! \brief 声明用于接收和发送的缓冲结构体
extern SQMat sqmat_recv;
extern SQMat sqmat_send;
//! \brief 声明用于标识接收数据类型的bool
extern SQMatStatus sqmat_enum_recv; 
extern bool isrecv_enum;

/**
 * \brief 默认单端口监听epoll服务端 
 * \brief 只连接到独一的qt客户端
*/
class QServer: public baseServer
{
public:
    using baseServer::_ctexts;
    using baseServer::_qtexts;
    using _eventflag = baseServer::_netflag;
    using _runflag = baseServer::_netflag;
    using baseServer::_size;
    using baseServer::_usize;
    using baseServer::_qtuint;
    using baseServer::_port;
    using baseServer::_sock;

    using _epollevent = epoll_event;
    using _veceves = std::vector<_epollevent>;
    using _vecpots = std::vector<_port>;
    using _lisev_thread = std::thread;
    using _ev_thread = std::thread;

    QServer() = delete; 
    //! \brief 传递本地ip 创建epoll底层
    explicit QServer(const _ctexts& serverip);

    ~QServer();

public:
    //! \brief 设置默认监听socket默认管理socket
    void setsocketdefault() override;
    
    //! \brief 设置默认监听等待事件 
    //! \brief 异步启动监听线程 通信线程
    void setepolleventdefault() override;

protected:
    //! \brief 异步启动 只处理监听就绪
    //! \brief 当默认listen socket关闭退出该线程
    void asyncepoll_lisevents_wait();

    //! \brief 异步启动 只处理cmd
    void asyncepoll_cmdevents_wait();

    //! \brief 异步启动 只处理通信send/recv
    void asyncepoll_events_wait();

private:
    //! \brief 解析命令并跳转
    void cmdanalysed(ServerCmd _cmd_recv);

    //! \brief 接收图像数据SQMat的函数
    bool server_socket_recv();

    //! \brief 发送推理结果SQMat的函数
    bool server_socket_sned();

protected:
    //! 创建epoll实例
    int _ser_epoll;

    //! 默认监听socket 默认管理socket 
    //! 注册为边缘触发 只读事件
    _sock _ser_lissocket;
    _sock _ser_cmdsocket;

    //! 默认监听连接成功后分配的通信socket
    _sock _ser_lissocket_com;
    _sock _ser_cmdsocket_com;

    //! 默认循环处理事件数目
    const _usize MAXEVENTS;
    //! 循环中就绪事件容器
    _veceves _events;

    //! 用于监听线程 
    _lisev_thread _async_lisev_t;
    //! 用于cmd事件处理线程 始终等待锁
    _ev_thread _async_cmdev_t;
    //! 用于事件处理线程 始终等待锁
    _ev_thread _async_ev_t;

    using _cmd_mutex = std::mutex;
    using _thread_cv = std::condition_variable;
    using _lock_cmd_mutex = std::unique_lock<std::mutex>;

    //! cmd命令线程持有锁并通知通信线程
    _thread_cv _cmd_cv;
    _cmd_mutex _cmd_mt;

private:
    //! 默认ip socket已设置标志
    _netflag _isipbind;
    //! 默认cmd listen通信线程运行标志
    _eventflag _iswaitcmdevent;
    _eventflag _iswaitevent;

protected:
    //! 被计算进程通知是否有数据发送
    _runflag _issend;

    //! 通知到计算进程推理状态信号 只有cmd设定trtset后为true
    _runflag _isrun;
};

#endif

#ifndef ReServer
#define ReServer
/**
 * \brief 增加端口监听epoll服务端 
*/
class DuServer: public QServer
{
public:
    using QServer::_ctexts;
    using QServer::_qtexts;
    using QServer::_eventflag;
    using QServer::_size;
    using QServer::_usize;
    using QServer::_qtuint;
    using QServer::_port;
    using QServer::_sock;
    using QServer::_veceves;
    using QServer::_vecpots;
    using QServer::_epollevent;
    using _vecsoks = std::vector<_sock>;

public:
    //! \brief 默认单端口
    DuServer(_ctexts serverip);
    //! \brief 设置更多端口
    explicit DuServer(_ctexts serverip, _vecpots or_ports);

    ~DuServer();

    //! \brief 设置默认监听等待事件 
    //! \brief 异步启动监听线程 通信线程 
    //! \brief 始终等待终端信号调用insertsockets/insertkeepepollwait
    //void setepolleventdefault() override;

    //! \brief 动态创建监听socket增加到epoll等待队列中
    //void insertsockets();

    //! \brief 动态增加epoll等待队列数目
    //void insertkeepepollwait(_usize new_evnet_num);

    //! 增加的监听socket端口容器
    _vecsoks or_lissockets;
    _vecpots or_ports;

    //! 额外循环处理事件数目
    const _usize NEWMAXEVENTS;
    //! 循环中额外就绪事件容器
    _veceves _newevents;

protected:
};

#endif

#ifndef Process
#define Process
//! \brief 特例化进程数量
#define NUMPROCESS  3 

/**
 * \brief 进程并发对象
*/
template<uint NUMFORKS>
class baseProcess
{
public:
    using _ctexts = typename TypeAlias::_ctexts;
    using _usize = typename TypeAlias::_usize;
    using _size = typename TypeAlias::_size;

    using sem_coll = int;        
    using sem_idx = int;

    using pid_vec = std::vector<pid_t>;
    using sem_vec = std::vector<sem_idx>;
    // 进程锁用于共享内存
    using pro_mutex = pthread_mutex_t;
    
    //! \brief 初始化调用
    baseProcess();
 
    virtual ~baseProcess();

    //! \brief 指定的进程添加到到kill
    void processkill(pid_t i) noexcept;

    //! \brief 启动所有子进程并等待信号量通知
    virtual int processinitwait();

    using _pro_func = std::function<void()>;
    using _base_func_p = void (baseProcess::*)(); 
    //! \brief 参数绑定到当前类成员函数并返回可执行封装函数
    //! \param 成员函数指针
    virtual _pro_func processbind(_base_func_p func);

    //! \brief 阻塞等待或是异步通知信号量
    //! \param 信号量集合 信号量索引 对应的sembuf结构体
    void processturn(sem_coll ids, sem_idx id, bool iswait=true) const;

    //! \brief 等待所有子进程
    void processwait() const;

public:
    //! 信号量集合
    sem_coll sem_collects;

    //! 子进程pid容器
    pid_vec _propidvec;
    //! 子进程信号量索引与对应的pid
    sem_vec _prosemvec;
    //! 结束后需要kill的进程pid
    pid_vec _propidkillvec;

    //! 记录已绑定子进程函数数目
    sem_idx _probindsum;
    //! 记录已创建子进程函数数目
    sem_idx _proforksum;
};

/**
 * \brief 包括qt界面进程 
*/
template<uint NUMFORKS>
class QProcess: virtual public baseProcess<NUMFORKS>
{
public:
    using typename baseProcess<NUMFORKS>::_ctexts;
    using typename baseProcess<NUMFORKS>::_usize;
    using typename baseProcess<NUMFORKS>::_size;
    using typename baseProcess<NUMFORKS>::sem_coll;
    using typename baseProcess<NUMFORKS>::sem_idx;

    using typename baseProcess<NUMFORKS>::pid_vec;
    using typename baseProcess<NUMFORKS>::sem_vec;
    using baseProcess<NUMFORKS>::pro_mutex;

    QProcess() = default;

    ~QProcess() = default;

    using typename baseProcess<NUMFORKS>::_pro_func;

    //! \brief 重声明虚函数
    using baseProcess<NUMFORKS>::processinitwait;
    //! \brief 重载启动函数 
    int processinitwait(_pro_func profunc);

    //! \brief 重声明虚函数
    using baseProcess<NUMFORKS>::processbind;
    //! \brief 重载绑定函数 
    using _QPro_func_p = void (QProcess::*)(const sem_idx&, const _ctexts&); 
    _pro_func processbind(_QPro_func_p func, _ctexts& s);
    
    //! \brief gui界面进程函数
    void process_qtgui(const sem_idx& sid, const _ctexts& ip);

    //! \brief epoll服务端进程函数
    void process_epollserver(const sem_idx& sid, const _ctexts& ip);

protected:
    //! qt界面指针
    SplashScreen *qtgui=nullptr;

    //! epoll对象指针
    QServer *server=nullptr;
};

/**
 * \brief 包括计算进程 
*/
template<uint NUMFORKS>
class RTProcess: virtual public QProcess<NUMFORKS>
{
public:
    using typename baseProcess<NUMFORKS>::_ctexts;
    using typename baseProcess<NUMFORKS>::_usize;
    using typename baseProcess<NUMFORKS>::_size;
    using typename baseProcess<NUMFORKS>::sem_coll;
    using typename baseProcess<NUMFORKS>::sem_idx;

    using typename baseProcess<NUMFORKS>::pid_vec;
    using typename baseProcess<NUMFORKS>::sem_vec;    

    RTProcess() = default;

    ~RTProcess() = default;

    using typename baseProcess<NUMFORKS>::_pro_func;

    //! 重声明processinitwait
    using baseProcess<NUMFORKS>::processinitwait;
    using QProcess<NUMPROCESS>::processinitwait;

    //! \brief 重声明绑定函数
    using QProcess<NUMPROCESS>::processbind;
    //! \brief 重载绑定函数
    using _RPro_func_p = void (RTProcess::*)(const sem_idx&, const _ctexts&); 
    _pro_func processbind(_RPro_func_p func, _ctexts& s);

    //! \brief 计算进程函数
    void process_trtrun(const sem_idx& sid, const _ctexts& ip);

protected:
    //! 计算线程池对象
    rtts<THREADS> *inference;
};

#endif
