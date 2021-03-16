# IRBuild

#### 介绍
（1）使用hiai foundation IR接口完成deconv算子的图构建，以及模型执行。
（2）实现C化接口，可以通过dlopen动态链接，从而节省加载内存

#### 软件架构
- test是可执行程序，只需要动态加载libXNN_NPU。

- libXNN_NPU是一个adapter，它依赖hiai的三个so，实现编译+执行一个deconv单算子模型。


#### 安装教程

1. 下载NDK工具，android-ndk-r20b及以上版本。
2. export PATH=$PATH:/Path_to/android-ndk-r20b/。

#### 使用说明

1.  编译脚本：build.sh
2.  执行脚本：push_run.sh

注意：

仅支持华为NPU手机。

