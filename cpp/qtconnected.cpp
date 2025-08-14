#include "qtpacked.h"
#include "rtpacketed.h"
//
//! ProcessShare对象
ProcessShare *ProcessShare::init()
{
    key_t key=ftok("/tmp", 6);
    //创建共享内存
    int share_id=shmget(key, sizeof(ProcessShare), SHARECREATE);
    //获取内存映射
    auto PS=reinterpret_cast<ProcessShare*>(shmat(share_id, nullptr, 0));
    //初始化成员与进程锁
    PS->share_id=share_id;
    PS->ref_count=0;
    PS->cache.reset();
    PS->iscache.store(false);

    pthread_mutexattr_t attr;
    assert(pthread_mutexattr_init(&attr) == 0);
    assert(pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) == 0);
    assert(pthread_mutex_init(&PS->send_mt, &attr) == 0);
    pthread_mutexattr_destroy(&attr);
    PS->ref_count++;
    //初始化信号量集合
    PS->sem_cache=semget(key, 1, SEMGET_CREATE);
    assert(PS->sem_cache != -1);
    semctl(PS->sem_cache, 0, SEMCTL_SET, 0);

    return PS;
}

ProcessShare *ProcessShare::get()
{
    key_t key=ftok("/tmp", 6);
    int share_id=shmget(key, sizeof(ProcessShare), SHAREGET);
    //获取内存映射
    auto PS=reinterpret_cast<ProcessShare*>(shmat(share_id, nullptr, 0));
    //引用次数
    pthread_mutex_lock(&PS->send_mt);
    PS->ref_count++;
    pthread_mutex_unlock(&PS->send_mt);
    return PS;
}

void ProcessShare::destroy()
{
    //引用次数
    pthread_mutex_lock(&this->send_mt);
    this->ref_count--;
    pthread_mutex_unlock(&this->send_mt);

    //引用0则销毁
    if (this->ref_count > 0) 
    { shmdt(this); return ; }

    semctl(this->sem_cache, 0, SEMCTL_REMOVE);

    pthread_mutex_destroy(&this->send_mt);
    shmctl(this->share_id, SHAREREMOVE, nullptr);
    std::cout<<"进程间对象已销毁\n";
    shmdt(this);
    return ;
}

TypeAlias::_status ProcessShare::push(SQMat& _push) {
    sembuf sb;
    sb.sem_num = 0;
    sb.sem_flg = 0; 
    sb.sem_op = -1;  // 等待"可写入"通知（来自pop的唤醒）

    // 若缓存有效（已有数据），必须等待pop取走并发送通知（信号量>0）
    if (this->iscache.load()) {
        // 此处阻塞，直到pop执行sem_op=1（信号量从1→0）
        if (semop(this->sem_cache, &sb, 1) != 0) {
            std::cerr << "push等待pop唤醒失败\n";
            return -1;
        }
    }
    
    // 加锁写入数据（此时iscache必为false）
    pthread_mutex_lock(&this->send_mt);
    this->cache.deepcopy(_push);
    this->iscache.store(true);  // 标记有数据
    pthread_mutex_unlock(&this->send_mt);

    // 通知pop：有数据可读取（信号量从0→1）
    sb.sem_op = 1;
    if (semop(this->sem_cache, &sb, 1) != 0) {
        std::cerr << "push通知pop失败\n";
        return -1;
    }
    std::cout << "push成功：" << _push.matpath << "，已通知pop\n";
    return 1;
}
//
//! exitscreen对象
exitscreen::exitscreen(bool& checked, QWidget *parent): 
QDialog(parent), exitdialogui(new Ui::exitDialog)
{
    this->exitdialogui->setupUi(this);
    this->checked=checked;
}
exitscreen::~exitscreen()
{
    delete this->exitdialogui;
}
void exitscreen::on_pushButton_clicked() 
{ this->checked=true; this->close(); }
//
//! talkscreen对象
talkscreen::talkscreen(_qtuint& numcontexts, _qtuint& numbatchsize, QWidget *parent): 
QWidget(parent), talkformui(new Ui::TalkForm)
{
    qDebug()<<"构造 talkformui成员";
    qDebug()<<"setup talkformui成员";

    this->talkformui->setupUi(this);
    qDebug()<<"现在开始记录推理过程";

    //设置连接状态
    this->iscontected.store(false);
    //设置推理状态
    //this->isrun.store(true);

    //设定来自初始页面的上下文 批量信息
    this->cnum=numcontexts;
    this->bnum=numbatchsize;
    qDebug()<<"确认分支界面talkformui设定 "<<"number_contexts="<<this->cnum;
    qDebug()<<"确认分支界面talkformui设定 "<<"number_batchsize="<<this->bnum;
    qDebug()<<"已准备分支界面talkformui";

    qDebug()<<"当前未连接server";
}
talkscreen::~talkscreen()
{
    ::close(this->cus_cmd_socket);
    ::close(this->cus_socket);
    //标识已断开连接状态
    this->iscontected.store(false);
    //析构SQMat序列
    for (auto& e: sqmat_vec)
        e.reset();
    //关闭页面
    delete this->talkformui;
}

void talkscreen::on_exitbutton_clicked()
{
    qDebug()<<"客户端断开连接 !";

    qDebug()<<"\n\n触发事件 是否退出qt界面程序";
    bool checked=false;
    //构造exitscreen对象
    this->exitdialogui=new exitscreen(checked);
    //阻塞等待
    this->exitdialogui->exec();

    checked=this->exitdialogui->checked;
    delete this->exitdialogui;
    if (!checked) {
        qDebug()<<"弹出窗口确认取消退出qt程序";
        return ;
    }
    qDebug()<<"弹出窗口确认当前退出qt程序 !";

    //尝试断开连接
    this->serverclose();

    sleep(2);
    qApp->quit();

    return ;
}

void talkscreen::on_contectbutton_clicked()
{
    qDebug()<<"\n\n触发事件 准备建立tcp/ip链接";
    if (this->iscontected.load()) {
        qDebug()<<"已经建立链接 "<<"无需再次链接";    
        return ;
    }

    //获取窗口输入 不检查直接用于连接
    _qtexts ipinfo=this->talkformui->iplineedit->text();
    _qtexts portinfo=this->talkformui->portlineedit->text();
    _qtexts cmdportinfo=this->talkformui->cmdportlineedit->text();

    this->ser_ip=ipinfo.toStdString();
    this->ser_port=portinfo.toUInt();
    this->ser_cmd_port=cmdportinfo.toUInt();

    //首次点击尝试链接
    std::cout<<"准备建立链接 "<<"确认ip地址为 "<<this->ser_ip<<"\n";
    std::cout<<"确认通信端口为 "<<this->ser_port<<"\n";
    std::cout<<"确认cmd端口为 "<<this->ser_cmd_port<<"\n";

    //尝试建立连接
    //建立通信socket 管理socket并测试
    this->serverconnected();
    //连接管理端口 
    this->servercmdconnected();

    //检查是否成功建立listen和cmd连接
    if (!this->iscontected.load()) {
        qDebug()<<"连接失败 !";

        ::close(this->cus_socket);
        ::close(this->cus_cmd_socket);
        return ;
    }

    //显式设置socket为非阻塞模式
    qDebug()<<"设置客户端socket和管理socket为非阻塞模式";
    int cus_socket_flag=::fcntl(this->cus_socket, SOCKFLAGGET, 0); //获取
    int cus_cmd_socket_flag=::fcntl(this->cus_cmd_socket, SOCKFLAGGET, 0);

    SOCKFLAGNOBLOCK(cus_socket_flag);
    SOCKFLAGNOBLOCK(cus_cmd_socket_flag);

    //重新设定
    if (::fcntl(this->cus_socket, SOCKFLAGSET, cus_socket_flag) < 0) {
        qDebug()<<"qt设置listen socket为非阻塞模式失败 ! 关闭连接"<<strerror(errno);
        ::close(this->cus_socket);
        this->iscontected.store(false);
        return ;
    }
    if (::fcntl(this->cus_cmd_socket, SOCKFLAGSET, cus_cmd_socket_flag) < 0) {
        qDebug()<<"qt设置cmd socket为非阻塞模式失败 ! 关闭连接"<<strerror(errno);
        ::close(this->cus_cmd_socket);
        this->iscontected.store(false);
        return ;
    }

    qDebug()<<"qt已完成默认连接";

    //发送设定推理信息命令 传递context batchsize信息
    //this->blockcmdsend(ServerCmd::TRTSET, this->cus_cmd_socket);
    //qDebug()<<"qt已发送初始trtset命令";

    //开启守护线程用于异步接收数据
    //std::thread t(&talkscreen::asyncreceive, this);
    //t.detach();

    //创建共享对象
    //this->_share=ProcessShare::init();
    this->inference=new rtts<THREADS>(CONTEXTS);

    this->inference->init();
    sleep(1);
    qDebug()<<"计算线程并发完毕\n";
    return ;
}

void talkscreen::serverconnected()
{
    //ipv4协议 流式数据
    this->cus_socket=::socket(IPV4, STREAMSR, 0);
    if (this->cus_socket < 0) {
        std::cerr<<"创建socket失败 !\n";
        return ;
    }

    //设置服务器地址对象
    sockaddr_in serveraddr{};
    serveraddr.sin_family=IPV4; //设置协议族
    serveraddr.sin_port=this->ser_port; //设置端口号
    //设置ip地址 将ip地址转为二进制模式
    inet_pton(IPV4, this->ser_ip.data(), &serveraddr.sin_addr.s_addr); 
    
    //无需bind connect时分配临时端口

    //connect模式 尝试连接
    int status=::connect(
        this->cus_socket, 
        reinterpret_cast<sockaddr*>(&serveraddr), 
        sizeof(serveraddr)
    );
    if (status < 0) {
        std::cerr<<"qt连接listen失败 !\n";
        ::close(this->cus_socket);
        return ;
    }

    std::cout<<"qt成功连接到listen端口\n";
    return ;
}

void talkscreen::servercmdconnected()
{
    //设置cmd socket为ipv4
    this->cus_cmd_socket=::socket(IPV4, STREAMSR, 0);
    if (this->cus_cmd_socket < 0) {
        std::cerr<<"创建cmd socket失败 !\n";
        return ;
    }

    //设置服务器cmd地址对象
    sockaddr_in servercmdaddr{};
    servercmdaddr.sin_family=IPV4;
    servercmdaddr.sin_port=this->ser_cmd_port;
    inet_pton(IPV4, this->ser_ip.data(), &servercmdaddr.sin_addr.s_addr);

    //无需bind connect时分配临时端口

    //尝试连接
    int status=::connect(
        this->cus_cmd_socket, 
        reinterpret_cast<sockaddr*>(&servercmdaddr), 
        sizeof(servercmdaddr)
    );
    if (status < 0) {
        std::cerr<<"qt连接cmd失败 !\n";
        ::close(this->cus_cmd_socket);
        this->iscontected.store(false);
        return ;
    }
    //更改状态标志
    this->iscontected.store(true);

    std::cout<<"qt成功连接到cmd端口\n";
    return ;
}

void talkscreen::serverclose()
{
    ::close(this->cus_cmd_socket);
    ::close(this->cus_socket);
    std::cout<<"已断开服务端连接\n";
    return ;
}
