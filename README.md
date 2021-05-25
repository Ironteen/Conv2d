#### 每天一个CUDA小技巧 ： 二维卷积的加速

> 题目：输入为一个512×512的矩阵，权重为3×3的矩阵，用"valid"的方式进行卷积，步长为1

##### 数据准备

- 使用numpy随机生成输入和权重的矩阵，并转为tensor，调用pytorch的F.conv2d计算结果，所有数据都保存为txt文件

  ```
  def generate_conv():
      inputs = np.round(np.random.rand(ih*iw), 4).astype(np.float32)
      np.savetxt("./inputs.txt", inputs, fmt="%s")
  
      weights = np.round(np.random.rand(kh*kw), 4).astype(np.float32)
      np.savetxt("./weights.txt", weights, fmt="%s")
  
      inputs = torch.tensor(np.reshape(inputs, [ih, iw])).view(1,1,ih,iw)
      weights = torch.tensor(np.reshape(weights, [kh, kw])).view(1,1,kh,kw)
      print(f"inputs : {inputs.size()}")
      print(f"weights : {weights.size()}")
  
      outputs = F.conv2d(inputs,weights)
      print(f"outputs : {outputs.size()}")
  
      outputs = outputs.view(-1).numpy()
      np.savetxt("./outputs.txt", outputs, fmt="%s")
  ```

- 读取输入和权重文件，再次计算卷积结果，并与txt文件进行对比验证，确保无误

  ```
  def load_conv():
      inputs = np.loadtxt("./inputs.txt", np.float32)
      weights = np.loadtxt("./weights.txt", np.float32)
      outputs_gt = np.loadtxt("./outputs.txt", np.float32)
  
      inputs = torch.tensor(np.reshape(inputs, [ih, iw])).view(1,1,ih,iw)
      weights = torch.tensor(np.reshape(weights, [kh, kw])).view(1,1,kh,kw)
      print(f"inputs : {inputs.size()}")
      print(f"weights : {weights.size()}")
  
      start = time.time()
      outputs = F.conv2d(inputs,weights)
      end = time.time()
      print(f"duration : {end-start} second")
      print(f"outputs : {outputs.size()}")
  
      outputs = outputs.view(-1).numpy()
      diff = np.sum(outputs_gt-outputs)
      print(f"diff : {diff}")
  ```

##### CPU实现

- 思路：滑动窗 + 每个窗内进行For循环

- 步骤：

  - 分配内存三个矩阵的值 - 输入，权重，输出，并载入数据

    ```
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
    ```

    其中载入数据的函数如下：

    ```
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
    ```

  - 滑动窗卷积

    ```
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
    
    Conv2d_CPU(Inputs_host,Weights_host,Outputs_host,ih,iw,kh,kw);
    ```

  - 释放内存

    ```
    cudaFree(Inputs_dev);
    cudaFree(Weights_dev);
    cudaFree(Outputs_dev);
    cudaFree(Outputs_py);
    ```

##### CUDA实现

- 思路：每个滑动窗单独由一个线程来计算，所有滑动窗的卷积计算并行

- 步骤：

  - 分配内存，从主机(host) 向 显卡 (device)搬运原始数据

    ```
    float *Inputs_dev, *Weights_dev, *Outputs_dev;
    cudaMalloc((void**)&Inputs_dev,sizeof(float)  * ifm_nBytes);
    cudaMalloc((void**)&Weights_dev,sizeof(float) * wt_nBytes);
    cudaMalloc((void**)&Outputs_dev,sizeof(float) * ofm_nBytes);
    
    float* Outputs_from_gpu = (float*)malloc(sizeof(float) * ofm_nBytes);
    
    cudaMemcpy(Inputs_dev,Inputs_host,ifm_nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(Weights_dev,Weights_host,wt_nBytes,cudaMemcpyHostToDevice);
    ```

  - 设置并行数，分配线程

    ```
    int dimx=1, dimy=1;
    
    // 2d block and 2d grid
    dim3 block_0(dimx,dimy);
    dim3 grid_0(ih-2, iw-2);
    ```

    输入为ih×iw，3×3的卷积在步长为1，不对输入进行padding的情况下，需要进行(ih-2)×(iw-2)次卷积计算，我们这里让每个线程单独去负责一个窗口的乘累加计算

  - 启动核函数，进行计算，并进行同步

    ```
    __global__ void Conv2d_GPU(float * ifm,float * wt,float * ofm,int ih,int iw, int kh, int kw){
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
    
    Conv2d_GPU<<<grid_0,block_0>>>(Inputs_dev, Weights_dev, Outputs_dev, ih, iw, kh, kw);
    cudaDeviceSynchronize();
    ```

  - 将计算结果从显卡 (device)搬运回主机(host)

    ```
    cudaMemcpy(Outputs_from_gpu, Outputs_dev,ofm_nBytes,cudaMemcpyDeviceToHost);
    ```

  - 释放内存，重置 CUDA

    ```
    cudaFree(Inputs_dev);
    cudaFree(Weights_dev);
    cudaFree(Outputs_dev);
    cudaFree(Outputs_py);
    
    cudaDeviceReset();
    ```

#### 检查结果并时间对比

```
Using device 0: GeForce RTX 2080 Ti
CPU Execution Time elapsed 0.015028 sec
grid : (510, 510), block : (1, 1)
GPU Execution configuration<<<(510,510),(1,1)>>> Time elapsed 0.000339 sec
Acceleration of GPU to CPU : 44.326302
```

**加速比** ： 44.33



