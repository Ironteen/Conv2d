#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

#include "utils.h"


using namespace std;

void LoadData(float * data, string path){
	ifstream ifstr_data(path);
	float d;
    int i=0;
	while (ifstr_data >> d){
        data[i] = (float)d;
        i++;
    }
	ifstr_data.close();
}

void WriteData(float * data, int N, string path){
    ofstream ofile;          
    ofile.open(path);
    for(int i=0; i<N; i++)
        ofile<<data[i]<<endl;
    ofile.close();
}

void Conv2d_CPU(float * ifm,float * wt,float * ofm,int ih,int iw, int kh, int kw){
    int ind1, ind2, ind3;
    for(int ih1=1; ih1<ih-1; ih1++)
        for(int iw1=1; iw1<iw-1; iw1++){
            float tmp = 0;
            for(int kh1=0; kh1<kh; kh1++)
                for(int kw1=0; kw1<kw; kw1++){
                    ind1 = (ih1+kh1-1)*iw + iw1+kw1-1;
                    ind2 = kh1*kw + kw1;
                    tmp += ifm[ind1] * wt[ind2];
                }
            ind3 = (ih1-1)*(iw-2) + iw1-1;
            ofm[ind3] = tmp;
        }
}

__global__ void Conv2d_GPU(float * ifm,float * wt,float * ofm,int ih,int iw, int kh, int kw)
{
    float tmp = 0;
    int ind1, ind2, ind3;
    for(int i=0; i<kh; i++)
        for(int j=0; j<kw; j++){
            ind1 = (blockIdx.x+i)*iw + blockIdx.y+j;
            ind2 = i*kw + j;
            tmp += ifm[ind1] * wt[ind2];
        }
    ind3 = blockIdx.x*(iw-2) + blockIdx.y;
    ofm[ind3] = tmp;
}

int main(){
    // init devices
    initDevice(0);

    int ih=1<<9, iw=1<<9;
    int kh=3, kw=3;
    string inputs_path  = "./inputs.txt";
    string weights_path = "./weights.txt";
    string outputs_path = "./outputs.txt";

    int ifm_nBytes = ih*iw*sizeof(float);
    int wt_nBytes  = kh*kw*sizeof(float);
    int ofm_nBytes = (ih-2)*(iw-2)*sizeof(float);

    //Malloc
    float* Inputs_host  = (float*)malloc(sizeof(float) * ifm_nBytes);
    float* Weights_host = (float*)malloc(sizeof(float) * wt_nBytes);
    float* Outputs_host = (float*)malloc(sizeof(float) * ofm_nBytes);
    float* Outputs_py = (float*)malloc(sizeof(float) * ofm_nBytes);

    LoadData(Inputs_host, inputs_path);
    LoadData(Weights_host, weights_path);
    LoadData(Outputs_py, outputs_path);

    // cpu compute
    double iStart=cpuSecond();

    Conv2d_CPU(Inputs_host,Weights_host,Outputs_host,ih,iw,kh,kw);

    double iElaps_CPU=cpuSecond()-iStart;
    printf("CPU Execution Time elapsed %f sec\n",iElaps_CPU);

    // check
    string save_path = "./outputs_CPU.txt";
    WriteData(Outputs_host, (ih-2)*(iw-2), save_path);

    //cudaMalloc
    float *Inputs_dev, *Weights_dev, *Outputs_dev;
    cudaMalloc((void**)&Inputs_dev,sizeof(float)  * ifm_nBytes);
    cudaMalloc((void**)&Weights_dev,sizeof(float) * wt_nBytes);
    cudaMalloc((void**)&Outputs_dev,sizeof(float) * ofm_nBytes);

    float* Outputs_from_gpu = (float*)malloc(sizeof(float) * ofm_nBytes);

    cudaMemcpy(Inputs_dev,Inputs_host,ifm_nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(Weights_dev,Weights_host,wt_nBytes,cudaMemcpyHostToDevice);

    int dimx=1, dimy=1;

    // 2d block and 2d grid
    dim3 block_0(dimx,dimy);
    dim3 grid_0(ih-2, iw-2);
    printf("grid : (%d, %d), block : (%d, %d)\n", grid_0.x, grid_0.y, block_0.x, block_0.y);

    iStart=cpuSecond();

    Conv2d_GPU<<<grid_0,block_0>>>(Inputs_dev, Weights_dev, Outputs_dev, ih, iw, kh, kw);
    cudaDeviceSynchronize();

    double iElaps_GPU=cpuSecond()-iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
        grid_0.x,grid_0.y,block_0.x,block_0.y,iElaps_GPU);

    cudaMemcpy(Outputs_from_gpu, Outputs_dev,ofm_nBytes,cudaMemcpyDeviceToHost);

    // check
    save_path = "../outputs_GPU.txt";
    WriteData(Outputs_from_gpu, (ih-2)*(iw-2), save_path);

    // compare
    float ratio = iElaps_CPU/iElaps_GPU;
    printf("Acceleration of GPU to CPU : %f\n", ratio);

    cudaFree(Inputs_dev);
    cudaFree(Weights_dev);
    cudaFree(Outputs_dev);
    cudaFree(Outputs_py);
    cudaDeviceReset();

    free(Outputs_from_gpu);
    free(Inputs_host);
    free(Weights_host);
    free(Outputs_host);

    return 0;
}
