#include "serverpacked.h"

//! QServer对象
QServer::QServer(const _ctexts& serverip): 
MAXEVENTS(DEFAULTEVENTS)
{
    //设置默认ip 监听端口
    this->_ser_ip=serverip;
    this->_ser_port=SERLISPORT;
    this->_ser_cmd_port=SERCMDPORT;
    std::cout<<"server默认服务ip设置为"<<this->_ser_ip<<"\n";
    std::cout<<"server默认listen监听端口设置为"<<this->_ser_port<<"\n";
    std::cout<<"server默认cmd端口设置为"<<this->_ser_cmd_port<<"\n";
    
    //创建epoll实例
    this->_ser_epoll=epoll_create1(0);
    assert(this->_ser_epoll >= 0);

    //默认状态
    this->_isipbind.store(false);
    this->_iswaitcmdevent.store(false);
    this->_iswaitevent.store(false);
    this->_issend.store(false);
    this->_isrun.store(false);
}

QServer::~QServer()
{
    //先关闭分离通信线程
    if (this->_iswaitcmdevent.load()) {
        std::cout<<"关闭服务端默认cmd通信 !\n";
        //设置标志离开循环
        this->_iswaitcmdevent.store(false);

        //close触发 接触wait阻塞
        ::close(this->_ser_cmdsocket_com);
    }

    if (this->_iswaitevent.load()) {
        std::cout<<"关闭服务端默认listen通信 !\n";
        //设置标志离开循环
        this->_iswaitevent.store(false);

        //close触发 接触wait阻塞
        ::close(this->_ser_lissocket_com);
    }

    //最后关闭监听ip
    if (this->_isipbind) {
        std::cout<<"关闭服务端默认监听socket !\n";
        //设置标志离开循环
        this->_isipbind.store(false);

        //close边缘触发 接触wait阻塞
        ::close(this->_ser_lissocket);
        ::close(this->_ser_cmdsocket);
    }
}

void QServer::setsocketdefault()
{
    this->_ser_lissocket=::socket(IPV4, STREAMSR, 0);
    this->_ser_cmdsocket=::socket(IPV4, STREAMSR, 0);
    assert(this->_ser_lissocket > 0);
    assert(this->_ser_cmdsocket > 0);

    //设置本地监听地址 cmd地址
    sockaddr_in ser_addr{};
    ser_addr.sin_family=IPV4;
    ser_addr.sin_port=this->_ser_port;
    ::inet_pton(IPV4, this->_ser_ip.data(), &ser_addr.sin_addr.s_addr);

    sockaddr_in ser_cmd_addr{};
    ser_cmd_addr.sin_family=IPV4;
    ser_cmd_addr.sin_port=this->_ser_cmd_port;
    ::inet_pton(IPV4, this->_ser_ip.data(), &ser_cmd_addr.sin_addr.s_addr);

    const sockaddr *ser_addr_p=reinterpret_cast<const sockaddr*>(&ser_addr);
    const sockaddr *ser_cmd_addr_p=reinterpret_cast<const sockaddr*>(&ser_cmd_addr);
    _usize ser_addr_size=sizeof(ser_addr);
    _usize ser_cmd_addr_size=sizeof(ser_cmd_addr);

    //listen模式需要bind本地ip
    //sockaddr结构体仅保持bind时生命周期即可
    assert(::bind(this->_ser_lissocket, 
        ser_addr_p, ser_addr_size) >= 0);
    assert(::bind(this->_ser_cmdsocket, 
        ser_cmd_addr_p, ser_cmd_addr_size) >= 0);

    //listen模式 等待连接
    assert(::listen(this->_ser_lissocket,
        LISSOCOPT) >= 0);
    assert(::listen(this->_ser_cmdsocket,
        LISSOCMIN) >= 0);

    //设置为非阻塞 
    int ser_lissoc_f=::fcntl(this->_ser_lissocket, SOCKFLAGGET, 0);
    int ser_cmdsoc_f=::fcntl(this->_ser_cmdsocket, SOCKFLAGGET, 0);

    SOCKFLAGNOBLOCK(ser_lissoc_f);
    SOCKFLAGNOBLOCK(ser_cmdsoc_f);

    //设定
    assert(::fcntl(this->_ser_lissocket, SOCKFLAGSET, ser_lissoc_f) >= 0);
    assert(::fcntl(this->_ser_cmdsocket, SOCKFLAGSET, ser_cmdsoc_f) >= 0);

    this->_isipbind.store(true);
    
    std::cout<<"server已创建listen cmd socket并并绑定到ip "<<this->_ser_ip<<"\n";
    return ;
}

void QServer::setepolleventdefault()
{
    //是否提前设置默认ip
    if (!this->_isipbind.load()) {
        this->setsocketdefault();
        this->setepolleventdefault();
    }
    //add epoll
    else {
        //使用socket信息创建epoll事件
        _epollevent ser_lisevent{};
        _epollevent ser_cmdevent{};

        //设置为水平触发 只读事件
        ser_lisevent.events=EVENTRECV;
        ser_cmdevent.events=EVENTRECV;

        ser_lisevent.data.fd=this->_ser_lissocket;
        ser_cmdevent.data.fd=this->_ser_cmdsocket;

        //注册到epoll
        assert(::epoll_ctl(this->_ser_epoll, 
            EPOLLADD, this->_ser_lissocket, &ser_lisevent) >= 0);
        assert(::epoll_ctl(this->_ser_epoll, 
            EPOLLADD, this->_ser_cmdsocket, &ser_cmdevent) >= 0);
    }

    //开辟事件容器
    assert(this->MAXEVENTS > 0);
    this->_events.resize(this->MAXEVENTS);

    //主线程始终阻塞 
    //等待异步线程循环
    this->_async_lisev_t=_lisev_thread(
        &QServer::asyncepoll_lisevents_wait, 
        this
    ); 

    this->_async_cmdev_t=_ev_thread(
        &QServer::asyncepoll_cmdevents_wait,
        this
    );

    this->_async_ev_t=_ev_thread(
        &QServer::asyncepoll_events_wait, 
        this
    );

    std::cout<<"已启动默认网络监听 默认cmd监听 就绪等待\n";
    std::cout<<"已启动默认通信socket的epoll 就绪等待\n";

    //通信线程分离 
    if (this->_async_ev_t.joinable())
        this->_async_ev_t.detach();
    if (this->_async_cmdev_t.joinable())
        this->_async_cmdev_t.detach();
    //监听线程分离
    if (this->_async_lisev_t.joinable())
        this->_async_lisev_t.detach();

    this->_isrun.store(true);
    return ;
}

void QServer::asyncepoll_lisevents_wait()
{
    assert(this->_events.size() > 0);

    //根据监听socket是否建立判断是否循环
    while (this->_isipbind.load()) 
    {
    //wait
    int num_events=::epoll_wait(
        this->_ser_epoll,
        this->_events.data(), //就绪事件容器
        this->MAXEVENTS, //每次循环处理最大就绪事件
        -1 //阻塞等待
    );

    //确保触发状态
    if (num_events < 0) {
        std::cerr<<"监听线程epoll wait异常! 终止服务端";
        exit(EXIT_FAILURE);
    }

    for(int n=0; n<num_events; n++) {
        //处理服务端cmd监听事件
        if (this->_events[n].data.fd == this->_ser_cmdsocket) {
            //确认默认cmd监听是否关闭（主动析构）
            if (this->_events[n].events & (EVENTCLOSE | EVENTERR)) {
                std::cerr<<"默认cmd监听socket已关闭 跳出分支\n";
                ::epoll_ctl(this->_ser_epoll, EPOLLDEL, this->_ser_cmdsocket, nullptr);

                goto label_turn_cmd_down;
            }

            //cmd连接为只读 
            if (this->_events[n].events & EVENTRECV) {
                //循环处理所有连接申请
                while (1) 
                {
                    //接收新连接
                    _sock new_cmd_sock=::accept(
                        this->_ser_cmdsocket,
                        nullptr, //不关注客户端sockaddr结构体 
                        nullptr
                    );

                    //当前返回socket检查
                    if (new_cmd_sock >= 0) {
                        this->_ser_cmdsocket_com=new_cmd_sock;

                        //保证NOBLOCK socket
                        int f=::fcntl(this->_ser_cmdsocket_com, SOCKFLAGGET, 0); 
                        SOCKFLAGNOBLOCK(f);
                        assert(::fcntl(this->_ser_cmdsocket_com, SOCKFLAGSET, f) >= 0);

                        //注册到epoll 可读水平触发
                        _epollevent new_cmd_event{};
                        new_cmd_event.data.fd=this->_ser_cmdsocket_com;
                        new_cmd_event.events=EVENTRECV;

                        if (::epoll_ctl(this->_ser_epoll, EPOLLADD,
                                this->_ser_cmdsocket_com, &new_cmd_event) >= 0) {
                            std::cout<<"server默认cmd监听成功连接\n";

                            //cmd线程进入循环标志位
                            this->_iswaitcmdevent.store(true);
                        }
                        else {
                            std::cerr<<"cmd监听accept接受连接失败 忽略本次连接\n";
                            continue;
                        }
                    }
                    else {
                        if (errno == ACCEPTNONE) {
                            std::cout<<"cmd监听accept已循环处理所有申请\n";
                            break; //无更多连接申请退出
                        }
                        else {
                            std::cerr<<"cmd监听accept接受连接失败 忽略本次连接\n";
                            continue;
                        }
                    } 
                } 
            }
            else 
                std::cerr<<"cmd监听就绪事件不符合EVENTRECV 忽略所有连接\n";
        }
        label_turn_cmd_down:;

        //处理服务端listen监听事件
        if (this->_events[n].data.fd == this->_ser_lissocket) {
            //确认默认listen是否关闭（主动析构）
            if (this->_events[n].events & (EVENTCLOSE | EVENTERR)) {
                std::cerr<<"默认listen监听socket已关闭 跳出分支\n";
                ::epoll_ctl(this->_ser_epoll, EPOLLDEL, this->_ser_lissocket, nullptr);

                goto label_turn_listen_down;
            }

            //listen连接为只读
            if (this->_events[n].events & EVENTRECV) {
                //循环处理所有连接申请
                while (1) 
                {
                    //接收新连接
                    _sock new_lis_sock=::accept(
                        this->_ser_lissocket,
                        nullptr, //不关注客户端sockaddr结构体 
                        nullptr
                    );

                    //当前返回socket检查
                    if (new_lis_sock >= 0) {
                        this->_ser_lissocket_com=new_lis_sock;

                        //保证NOBLOCK socket
                        int f=::fcntl(this->_ser_lissocket_com, SOCKFLAGGET, 0); 
                        SOCKFLAGNOBLOCK(f);
                        assert(::fcntl(this->_ser_lissocket_com, SOCKFLAGSET, f) >= 0);

                        //注册到epoll 可读水平触发
                        _epollevent new_lis_event{};
                        new_lis_event.data.fd=this->_ser_lissocket_com;
                        new_lis_event.events=EVENTRECV;

                        if (::epoll_ctl(this->_ser_epoll, EPOLLADD,
                                this->_ser_lissocket_com, &new_lis_event) >= 0) {
                            std::cout<<"server默认listen监听成功连接\n";

                            //通信线程进入循环
                            this->_iswaitevent.store(true);
                        }
                        else {
                            std::cerr<<"listen监听accept接受连接失败 忽略本次连接\n";
                            continue;;
                        }
                    }
                    else {
                        if (errno == ACCEPTNONE) {
                            std::cout<<"listen监听accept已循环处理所有申请\n";
                            break; //无更多连接申请退出
                        }
                        else {
                            std::cerr<<"listen监听accept接受连接失败 忽略本次连接\n";
                            continue;
                        }
                    } 
                } 
            }
            else          
                std::cerr<<"listen监听就绪事件不符合EVENTRECV 忽略所有连接\n";
        }
        label_turn_listen_down:;
    //for single down
    }

    //下一次等待所有就绪的监听事件
    continue;
    }

    //线程退出
    std::cout<<"退出监听线程\n";
    return ;
}

void QServer::asyncepoll_events_wait()
{
    assert(this->_events.size() > 0);

    //等待开启
    while (1) {
        if (this->_iswaitevent.load())
            break;
    }
    
    //除非主动析构后退出该线程 否则保持长期连接
    while (this->_iswaitevent.load())
    {
    //wait
    int num_events=::epoll_wait(
        this->_ser_epoll,
        this->_events.data(),
        this->MAXEVENTS,
        1000 //1s检查
    );
    //确保wait状态
    if (num_events == 0) {
        //==0说明1s检查无时间发生 回到下一次wait
        continue;
    }
    else if (num_events < 0) {
        switch (errno)
        {
        case EINTR:
            //==ENTIR说明系统调用中断 重新wait即可
            continue;
        
        default:
            std::cerr<<"listen通信线程epoll wait异常! 终止服务端";
            exit(EXIT_FAILURE);
        }
    }

    //>0说明有事件发生
    //对于recv事件 处理前等待可能得cmd接收事件完毕后解锁
    for (int n=0; n<num_events; n++) {
    //只处理通信事件
    if (this->_events[n].data.fd == this->_ser_lissocket_com) 
    {
        //close状态（主动析构） 
        if (this->_events[n].events & (EVENTCLOSE | EVENTERR)) {
            std::cerr<<"默认listen socket已关闭 跳出分支\n";
            ::epoll_ctl(this->_ser_epoll, EPOLLDEL, this->_ser_lissocket_com, nullptr);

            goto label_turn_down;
        }

        //如果计算进程_issend通知需要send 修改events状态
        if (this->_issend.load()) {
            _epollevent new_event{};
            new_event.data.fd=this->_ser_lissocket_com;
            new_event.events=EVENTRECV | EVENTSEND;
            //注册为可写同时可读 可能下一次写同时有读事件发生
            if (::epoll_ctl(this->_ser_epoll, EPOLLMOD, 
                this->_ser_lissocket_com, &new_event) < 0) {
                std::cerr<<"listen epoll事件修改为可写失败 无法发送推理数据 !\n";
            }
            //修改为可写 下一次wait立即返回
        }
        
        //这次for是recv类型
        if (this->_events[n].events & EVENTRECV) {
            //可能同时有cmd接收 等待被通知
            { 
            _lock_cmd_mutex lock(this->_cmd_mt);
            //瞬时解锁并等待
            this->_cmd_cv.wait(lock);
            }
            //接收SQMat 已内置检查
            assert(this->server_socket_recv());
        }
        //这次for是send类型
        else if (this->_events[n].events & EVENTSEND) {
            //因为_issend信号送达 立即发送
            //发送SQMat 已内置检查
            assert(this->server_socket_sned());
        }
        //意外情况 不是可读可写事件
        else {
            std::cerr<<"listen通信就绪事件不符合EVENTRECV或者EVENTSEND 忽略本次事件并修改水平触发可读模式\n";

            this->_events[n].events=EVENTRECV;
            if (::epoll_ctl(this->_ser_epoll, EPOLLMOD, 
                this->_ser_lissocket_com, &this->_events[n]) < 0) {
                std::cerr<<"listen epoll 修改可读失败 !\n";
            }
            goto label_turn_down;
        }
    }
    
    //for single down
    label_turn_down:;
    }
    
    //下一次等待就绪的事件
    continue;
    }
    
    //线程退出
    std::cout<<"退出listen socket通信线程\n";
    return ;
}

//! DuServer对象
DuServer::DuServer(_ctexts serverip): QServer(serverip), NEWMAXEVENTS(DEFAULTEVENTS)
{}
DuServer::DuServer(_ctexts serverip, _vecpots or_ports): QServer(serverip), NEWMAXEVENTS(or_ports.size())
{
    this->or_ports.resize(or_ports.size());
    std::copy(or_ports.begin(), or_ports.end(), this->or_ports.begin());
}

DuServer::~DuServer()
{
    //关闭额外监听ip
    //if (this->_isipbind) {
    //    std::cout<<"关闭服务端额外监听socket !\n";
    //    if (this->or_lissockets.size() <= 0)
    //        return ;
    //    for (auto& i: this->or_lissockets)
    //        ::close(i);
    //}
    //_isipbind不改变状态 
}
