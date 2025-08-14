#include "qtpacked.h"
#include "serverpacked.h"
#include "rtpacketed.h"
//
//! process对象
template<>
baseProcess<NUMPROCESS>::baseProcess()
{
    //生成唯一key
    key_t key = ftok("/tmp", 66);
    if (key == -1) {
        perror("ftok failed");  
        exit(EXIT_FAILURE);
    }

    //创建信号量集合
    this->sem_collects=semget(key, NUMPROCESS, SEMGET_CREATE);
    if (this->sem_collects == -1) {
        perror("semget failed");  
        exit(EXIT_FAILURE);
    }

    //初始化信号量
    for (sem_idx i=0; i<NUMPROCESS; i++) {
        int s=semctl(this->sem_collects, i, SEMCTL_SET, 0);
        assert(s != -1);
    }

    //开辟容器
    this->_propidvec.reserve(NUMPROCESS);
    this->_propidkillvec.reserve(NUMPROCESS);
    this->_prosemvec.reserve(NUMPROCESS);
    for (sem_idx i=0; i<NUMPROCESS; i++) {
        this->_propidvec.push_back(-1);
        this->_propidkillvec.push_back(-1);
        this->_prosemvec.push_back(i);
    }

    this->_probindsum=0;
    this->_proforksum=0;
    std::cout<<"父进程pid "<<getpid()<<"\n";
}

template<>
baseProcess<NUMPROCESS>::~baseProcess()
{
    //删除信号量集合
    semctl(sem_collects, 0, SEMCTL_REMOVE);
    //等待子线程
    for (pid_t& i: this->_propidvec) {
    if (i != -1)
        waitpid(i, nullptr, 0);
    }
    //强制kill
    for (pid_t& i: this->_propidkillvec) {
    if (i != -1)
        kill(i, SIGKILL);
    }
    exit(EXIT_SUCCESS);
}

template<>
void baseProcess<NUMPROCESS>::processkill(pid_t i) noexcept
{
    for (auto& p: this->_propidkillvec) {
    if (p == -1)
        continue;
    //该位置覆盖
    p=i;
    break;
    }
    return ;
}

template<>
void baseProcess<NUMPROCESS>::processturn(sem_coll ids, sem_idx id, bool iswait) const
{
    //临时构造信号量操作结构体
    sembuf op;
    op.sem_num=id;
    op.sem_flg=0;
    //wait
    if (iswait) {
        op.sem_op=-1;
        assert(::semop(ids, &op, 1) != -1);
    }
    //notice
    else {
        op.sem_op=1;
        assert(::semop(ids, &op, 1) != -1);
    }
    return ;
}

template<>
baseProcess<NUMPROCESS>::_pro_func baseProcess<NUMPROCESS>::processbind(_base_func_p func)
{
    //是否可绑定
    if (this->_probindsum > NUMPROCESS)
        std::cout<<"绑定的子进程函数超出模版特例NUMPROCESS限定 !\n";
    //该进程函数对应信号索引
    const sem_idx& i=this->_prosemvec[this->_probindsum];

    //绑定当前对象
    //分配信号量索引
    _pro_func bind_func=std::bind(func, this);
    //记录绑定数量
    this->_probindsum++;

    return bind_func;
}

template<>
int baseProcess<NUMPROCESS>::processinitwait()
{
    std::cout<<"父进程pid "<<getpid()<<"\n";

    for (int i=0; i<NUMPROCESS; i++) {
    pid_t p=fork();
    
    if (p < 0) {
        std::cerr<<"子进程创建失败 !\n";
        exit(EXIT_FAILURE);
    }
    else if (p > 0) {
        //父进程作用域
        this->_propidvec[i]=p;
    }
    else if (p == 0) {
        //子进程作用域
        std::cout<<"子进程pid "<<getpid()<<"\n";
        ::sleep(2);

        //等待被通知
        sem_idx idx=i;
        this->processturn(this->sem_collects, idx);
        std::cout<<"子进程被唤醒 pid "<<getpid()<<"\n";\

        exit(EXIT_SUCCESS);
    }
    }

    std::cout<<"父进程创建完毕 在10s后唤醒所有子进程\n";
    std::this_thread::sleep_for(std::chrono::seconds(10));

    for (int i=0; i<NUMPROCESS; i++) {
        this->processturn(
            this->sem_collects, 
            this->_prosemvec[i],
            false
        );
    }  

    std::cout<<"父进程唤醒完毕\n";
    return 0;
}

template<>
void baseProcess<NUMPROCESS>::processwait() const 
{
    //阻塞等待
    for (const pid_t& i: this->_propidvec) {
        pid_t result=waitpid(i, nullptr, 0);
        if (result == -1) {
            continue;
        }
    }
    //父进程结束
    std::cout<<"\n父进程结束\n";
    return ;
}

//
//! QProcess对象
template<>
void QProcess<NUMPROCESS>::process_qtgui(const sem_idx& sid, const _ctexts& ip)
{
    //等待通知
    this->processturn(this->sem_collects, sid);

    qDebug()<<"\nqtgui进程启动 pid "<<getpid()<<'\n';
    
    int argc = 1;
    char* argv[] = { (char*)"qt_gui_process", nullptr }; 
    QApplication a(argc, argv);
    
    //构造qt指针对象
    this->qtgui=new SplashScreen();    

    // 创建并显示自定义UI窗口
    SplashScreen splash;
    splash.show();
    //循环刷新显示
    a.exec();
    
    //主动关闭界面
    this->qtgui=nullptr;
    qDebug()<<"qtgui进程结束";
    exit(EXIT_SUCCESS);
}

template<>
void QProcess<NUMPROCESS>::process_epollserver(const sem_idx& sid, const _ctexts& ip)
{
    //等待通知
    this->processturn(this->sem_collects, sid);

    std::cout<<"server进程启动 pid "<<getpid()<<'\n';
    std::cout<<"强制停止输入 down\n";
    //构造epollserver指针对象
    this->server=new QServer(ip);
    //设置socket
    this->server->setsocketdefault();
    //启动线程epollwait
    this->server->setepolleventdefault();
    
    //等待终端命令行 主动析构
    std::string s;
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::getline(std::cin, s);
        int b=strcmp(s.data(), "down");
        if (b != 0) {
            std::cout<<"输入命令未知 强制停止输入 down\n";
            continue;
        }
        else if (b == 0)
            break;
    }

    //退出epoll进程 析构
    delete this->server;
    std::cout<<"server进程结束\n";
    exit(EXIT_SUCCESS);
}

template<>
QProcess<NUMPROCESS>::_pro_func QProcess<NUMPROCESS>::processbind(_QPro_func_p func, _ctexts& s)
{
    //是否可绑定
    if (this->_probindsum > NUMPROCESS)
        std::cout<<"绑定的子进程函数超出模版特例NUMPROCESS限定 !\n";
    //该进程函数对应信号索引
    const sem_idx& i=this->_prosemvec[this->_probindsum];

    //绑定当前对象
    //分配信号量索引
    _pro_func bind_func=std::bind
    (func, this, std::ref(i), std::ref(s));
    //记录绑定数量
    this->_probindsum++;

    return bind_func;
}

template<>
int QProcess<NUMPROCESS>::processinitwait(_pro_func profunc)
{
    //是否fork
    if (this->_proforksum >= NUMPROCESS) {
        std::cerr<<"创建子进程已超出模版特例NUMPROCESS限定 !\n";
        return -1;
    }
    pid_t pd=fork();
    if (pd < 0) {
        std::cerr<<"创建子进程失败 已停止\n";
        exit(1);
    }
    else if (pd > 0) {
        //容器记录pid
        this->_propidvec[this->_proforksum]=pd;
    }
    else if (pd == 0) {
        //启动子线程
        profunc();
        //子线程结束
        exit(EXIT_SUCCESS);
    }
    //父进程返回
    this->_proforksum++;
    return 1;
}
//
//!
template<>
void RTProcess<NUMPROCESS>::process_trtrun(const sem_idx& sid, const _ctexts& ip)
{
    //等待通知
    this->processturn(this->sem_collects, sid);

    std::cout<<"计算进程启动 pid "<<getpid()<<'\n';
    
    //构造线程池对象
    this->inference=new rtts<THREADS>(CONTEXTS);
    //阻塞工作
    this->inference->init();

    delete this->inference;
    qDebug()<<"计算进程结束";
    exit(EXIT_SUCCESS);
}

template<>
RTProcess<NUMPROCESS>::_pro_func RTProcess<NUMPROCESS>::processbind(_RPro_func_p func, _ctexts& s)
{
    //是否可绑定
    if (this->_probindsum > NUMPROCESS)
        std::cout<<"绑定的子进程函数超出模版特例NUMPROCESS限定 !\n";
    //该进程函数对应信号索引
    const sem_idx& i=this->_prosemvec[this->_probindsum];

    //绑定当前对象
    //分配信号量索引
    _pro_func bind_func=std::bind
    (func, this, std::ref(i), std::ref(s));
    //记录绑定数量
    this->_probindsum++;

    return bind_func;
}