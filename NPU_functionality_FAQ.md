# NPU FUNCTIONALITY FAQ

​		本章我们给出使用HIAI foundation过程中场景问题的处理方法。

###  场景一：NPU推理数据如何与标杆数据做比对？

​		npu内部采用FP16计算，计算结果存在与FP32标杆结果间的误差。那我们应该通过什么手段来衡量误差的多少？是否应用具体算法的部署呢？这里我们给出我们内部业务实际部署的时候的评价标准。

> 下面会通过python3脚本的方式给大家提供对应的精度评价函数，这里首先请大家安装python3.7，以及对应的numpy工具包
>
> python下载链接：https://www.python.org/downloads/
>
> 安装numpy： `pip3 install numpy`

##### 读取和写入的数据文件

```python
import numpy as np
import struct
import os

def load_bin(filepath, dtype = np.float32) : 
    with open(filepath, 'rb') as f:
        content = f.read();
    data = np.frombuffer(content,dtype=dtype)
    return data

def save_bin(output,filepath):
    with open(filepath,'wb') as f :
        f.write(output.tobytes())
        f.close()
    return

#usage example
result = load_bin('/Path_to_data/npu.bin',np.float32)
print(result)

input1 = np.ones((1,3,224,224),np.float32)
save_bin(input1)
```

##### 常见文件的比较方法

```python
import numpy as np
import struct
import os

# 加载数据函数
def load_bin(filepath, dtype = np.float32) : 
    with open(filepath, 'rb') as f:
        content = f.read();
    data = np.frombuffer(content,dtype=dtype)
    return data
  
# 找出数据中的最大5个值对应的索引
def top5(input):
    tmp = input.reshape(-1)
    ind = np.argpartition(tmp, -5)[-5:]
    top = ind[::-1]
    print("*****************Top5*****************")
    print(top[0],tmp[top[0]])
    print(top[1],tmp[top[1]])
    print(top[2],tmp[top[2]])
    print(top[3],tmp[top[3]])
    print(top[4],tmp[top[4]])
    print("**************************************")
    print("")
    
# 计算数据向量的最大误差、平均误差、余弦相似度
def compare_cos(input1,input2) :
    input1 = np.reshape(input1,(-1))
    input2 = np.reshape(input2,(-1))
    print("最大误差：", np.max(np.abs(input1-input2)))
    print("平均误差：", np.mean(np.abs(input1-input2)))
    ab = np.sum(np.multiply(input1, input2))
    aa = np.sqrt(np.sum(np.multiply(input1, input1)))
    bb = np.sqrt(np.sum(np.multiply(input2, input2)))
    cos = ab/(aa*bb)
    print(cos)

# usage example
# 1.read data into numpy array
output1 = load_bin('/Path_to_data/npu.bin',np.float64)
output2 = load_bin('/Path_to_data/standard.bin',np.float64)
# 2.show top5 index of numpy array
top5(output1)
top5(output2)
# 3.calculate maximum error, mean error, cosine similarity 
compare_cos(output1, output2)
```



