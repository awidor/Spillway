#include "strata/program/pcie_auto.hpp"
#include <cstdlib>
#include <cmath>
#include <iostream>
using strata::program::PcieAuto;
static void require(bool passed) {
    if (!passed) {
        std::cerr << "pcie_auto invariant failed\n";
        std::exit(1);
    }
}
static double advance(PcieAuto& a, int n, double ms_step, int jobs_step, double cur) {
    double ms=a.mark_ms; int64_t jobs=a.mark_jobs; double answer=-1,tc=0,rate=0;
    for(int i=0;i<n;++i) { ms+=ms_step;jobs+=jobs_step;double f=a.window(ms,jobs,cur,&tc,&rate);if(f>=0)answer=f; }
    return answer;
}
int main() {
    PcieAuto a; a.link_gbps=12;a.blob_bytes=1000000;
    require(advance(a,60,10,10,.33)<0); // disabled leaves the fixed profile alone
    a.on=true;a.mark(900,900); // exclude prompt work from the first decode measurement
    require(advance(a,47,10,10,.33)<0);
    double f=advance(a,1,10,10,.33);require(f>=.10&&f<=.60);
    require(std::fabs(f*20-std::round(f*20))<1e-9); // bounded discrete changes
    advance(a,96,10,10,.33);advance(a,96,10,10,.33);
    require(a.updates==3&&!a.on); // cannot keep changing throughout a request
    PcieAuto hits;hits.on=true;hits.link_gbps=12;hits.blob_bytes=1000000;
    require(advance(hits,2000,0,0,.33)<0&&!hits.on); // cached experts must not produce a bogus CPU speed
    PcieAuto no_time;no_time.on=true;no_time.link_gbps=12;no_time.blob_bytes=1000000;
    require(advance(no_time,48,0,10,.33)<0); // zero measurement never divides by zero
    std::cout<<"pcie_auto: bounded updates, prompt exclusion, no-miss and zero-time cases passed\n";
}
