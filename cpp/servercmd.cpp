#include "serverpacked.h"

//! QServer对象
void QServer::asyncepoll_cmdevents_wait()
{
    assert(this->_events.size() > 0);

    //等待开启
    while (1) {
        if (this->_iswaitcmdevent.load())
            break;
    }
    
    //等待cmd命令
    //除非主动析构后退出该线程 否则保持长期连接
    while (this->_iswaitcmdevent.load())
    {
    //wait
    int num_events=::epoll_wait(
        this->_ser_epoll,
        this->_events.data(),
        this->MAXEVENTS,
        100 //0.1s超时检查
    );
    //每次wait后优先持有锁 避免多次锁定
    _lock_cmd_mutex lock(this->_cmd_mt, std::defer_lock);
    lock.lock();

    //查看wait状态
    if (num_events == 0) {
        //==0表明无就绪事件发生
        //解锁后返回 等待下一次检查
        lock.unlock();
        this->_cmd_cv.notify_one();
        continue;
    }
    else if (num_events < 0) {   
        switch (errno)
        {
        case EINTR:
            //==ENTIR说明系统调用中断 重新wait即可
            lock.unlock();
            this->_cmd_cv.notify_one();
            continue;
        
        default:
            std::cerr<<"cmd通信线程epoll wait异常! 终止服务端";
            exit(EXIT_FAILURE);
        }
    }

    //当>0说明有事件就绪 判断是否是cmd send
    //执行对应命令后解锁
    for (int n=0; n<num_events; n++) {
    //只处理命令事件
    if (this->_events[n].data.fd == this->_ser_cmdsocket_com) 
    {
        //close状态（主动析构）
        if (this->_events[n].events & (EVENTCLOSE | EVENTERR)) {
            std::cerr<<"默认cmd socket已关闭 跳出分支\n";
            ::epoll_ctl(this->_ser_epoll, EPOLLDEL, this->_ser_cmdsocket_com, nullptr);

            goto label_turn_cmd_down;
        }

        //cmd命令为只读
        if (this->_events[n].events & EVENTRECV) {
            std::cout<<"server wait触发有cmd命令等待接收\n";
            int icmd_recv;
            char *cmd_recv_p=reinterpret_cast<char*>(&icmd_recv);
            _usize cmd_recv_size_all=sizeof(icmd_recv);
            _usize cmd_recv_size=0;

            //循环接收命令
            while (cmd_recv_size < cmd_recv_size_all)
            {
                _size cn=::recv(
                    this->_ser_cmdsocket_com,
                    cmd_recv_p + cmd_recv_size,
                    sizeof(icmd_recv) - cmd_recv_size,
                    0
                );

                if (cn <= 0) {
                    if (errno == ACCEPTNONE)
                        break; //意外情况 已经无接收数据
                    else {
                        std::cerr<<"server cmd接收期间socket断开或其他错误 ! 终止服务端";
                        exit(EXIT_FAILURE);
                    }
                }

                cmd_recv_size+=cn;
            }

            //转换回枚举变量 执行命令
            ServerCmd cmd_recv=static_cast<ServerCmd>(icmd_recv);
            std::cout<<"server成功接收cmd命令 "<<ServerCmdToCString(cmd_recv)<<" 并执行\n";
            this->cmdanalysed(cmd_recv);
        }
        else {
            std::cerr<<"cmd命令就绪事件不符合EVENTRECV 忽略本次send的命令并修改水平触发可读模式\n";

            this->_events[n].events=EVENTRECV;
            if (::epoll_ctl(this->_ser_epoll, EPOLLMOD, 
                this->_ser_cmdsocket_com, &this->_events[n]) < 0) {
                std::cerr<<"cmd epoll 修改可读失败 !\n";
            }
            goto label_turn_cmd_down;
        }
    }

    //for single down
    label_turn_cmd_down:;
    }
    
    //wait前解锁
    lock.unlock();
    this->_cmd_cv.notify_one();
    //下一次等待就绪的命令
    continue;
    }
    
    //线程退出
    std::cout<<"退出默认cmd socket命令线程\n";
    return ;
}

void QServer::cmdanalysed(ServerCmd _cmd_recv)
{
    //对于trtset命令单独处理
    if (_cmd_recv == ServerCmd::TRTSET) {
        //等待一段时间确保数据到达 100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //接收设定
        _usize allsize=sizeof(uint) * 2;
        _usize recvsize=0;
        uint recvnum[2]={};
        char *recvnum_p=reinterpret_cast<char*>(recvnum);
        while (recvsize < allsize) {
            _size n=::recv(
                this->_ser_cmdsocket_com,
                recvnum_p + recvsize,
                allsize - recvsize,
                0
            );

            if (n <= 0) {
                if (errno == ACCEPTNONE)
                    break; //意外情况 已经无接收数据
                else {
                    std::cerr<<"server cmd trtset 接收期间socket断开或其他错误 ! 终止服务端";
                    exit(EXIT_FAILURE);
                }
            }

            recvsize+=n;            
        }
        std::cout<<"server cmd接收trtset设定 "<<
        "contexts "<<recvnum[0]<<" batchsize "<<recvnum[1]<<'\n';
        //设定传递到计算进程
        ////////////////////////////////////////////////    
        ////////////////////////////////////////////////
    }

    //其他命令
    switch (_cmd_recv)
    {
    case ServerCmd::STATUS:
        //打印服务器状态

        break;
    case ServerCmd::RECEIVEDOWN:
        //停止推理信号（不影响当前推理运行与主动send） 

        break;
    case ServerCmd::RECEIVEON:
        //继续推理信号

        break;
    case ServerCmd::TEST:
        //test
        std::cout<<"server cmd已接收到test命令\n";
        break;

    case ServerCmd::INVALID:
        return ;
    default:
        break;
    }
    return ;
}

