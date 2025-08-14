#include <iostream>
#include <string>
#include "qtpacked.h"
#include "serverpacked.h"
#include "rtpacketed.h"

int main(int argc, char *argv[])
{
    std::string s="";
    
    using _Process = RTProcess<NUMPROCESS>;
    //进程集合对象
    _Process process;
    //获取计算进程
    _Process::_pro_func f_rt=process.processbind(&_Process::process_trtrun, s);
    //根据当前绑定的进程函数记录获取信号量索引
    int f_rt_sem=process._prosemvec[process._probindsum - 1];

    //获取qt进程函数
    _Process::_pro_func f_qt=process.processbind(&_Process::process_qtgui, s);
    int f_qt_sem=process._prosemvec[process._probindsum - 1];

    //等待输入
    std::cout<<"本地 ip addr ";
    std::getline(std::cin, s);
    std::cout<<"确认 "<<s<<std::endl;

    //获取server进程函数
    _Process::_pro_func f_sv=process.processbind(&_Process::process_epollserver, s);
    int f_sv_sem=process._prosemvec[process._probindsum - 1];

    //启动server
    assert(process.processinitwait(f_sv) > 0);
    //立即通知
    process.processturn(process.sem_collects, f_sv_sem, false);

    using namespace std::this_thread;
    using namespace std::chrono;
    sleep_for(seconds(2));

    //启动qt
    assert(process.processinitwait(f_qt) > 0);
    //启动计算
    //assert(process.processinitwait(f_rt) > 0);

    //立即通知qt
    process.processturn(process.sem_collects, f_qt_sem, false);
    sleep_for(seconds(2));
    //通知计算
    //process.processturn(process.sem_collects, f_rt_sem, false);

    //父进程等待
    process.processwait();

    exit(0);
}
