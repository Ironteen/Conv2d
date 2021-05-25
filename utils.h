#ifndef UTILS_H
#define UTILS_H

#include <time.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#ifdef _WIN32

int gettimeofday(struct timeval *tp, void *tzp){
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year   = wtm.wYear - 1900;
    tm.tm_mon   = wtm.wMonth - 1;
    tm.tm_mday   = wtm.wDay;
    tm.tm_hour   = wtm.wHour;
    tm.tm_min   = wtm.wMinute;
    tm.tm_sec   = wtm.wSecond;
    tm. tm_isdst  = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

void initialData(float* ip,int size){
    time_t t;
    srand((unsigned )time(&t));
    for(int i=0;i<size;i++){
        ip[i]=(float)(rand()&0xffff)/1000.0f;
    }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,dev);
  printf("Using device %d: %s\n",dev,deviceProp.name);
  cudaSetDevice(dev);

}
void checkResult(float * hostRef,float * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}
#endif//FRESHMAN_H
