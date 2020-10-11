### 系统要求
libtorch：1.6.0+cuda10.1
cmake3.17
g++5.3-7.5都可使用
cuda：10.1

### 使用方法
1. 安装cmake3.17
2. 安装opencv-3.4.11
  - 下载opencv-3.4.11源码
  ```
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install/ -DBUILD_LIST=core,imgproc,imgcodecs ..
  make
  make install
  ```
3. git clone `this_project` && cd `this_project`
4. git clone c++的json库nlohmann_json，版本3.9.1(https://github.com/nlohmann/json)
5. 下载cpp模型`FasterRcnn_cpp.pt`和`ROIHead_cpp.pt`：https://pan.baidu.com/s/1AvrrhuhJz38vb66lRYmqcw ，提取码：ygel 。或者使用FasterRCNN.py和mmdetection2.2.1的模型文件自行生成。最后得到的文件目录如下图：
  ```
  |-- CMakeLists.txt
  |-- FasterRCNN.py
  |-- FasterRcnn_cpp.pt
  |-- README.md
  |-- ROIHead_cpp.pt
  |-- nms
  |   |-- CMakeLists.txt
  |   |-- nms_cuda.cu
  |   `-- nms_cuda.cuh
  |....
  |-- nlohmann_json
  |   |-- CMakeLists.txt
  |   |-- README.md
  |   |-- include
  |   |-- meson.build
  |   ....
  ```
6. 编译模型
  ```
  mkdir Release
  cd Release
  cmake3 -DCMAKE_PREFIX_PATH="/path/to/libtorch1.6.0/;/path/to/opencv/install"  ..  -DCMAKE_BUILD_TYPE=Release
  cmake3 --build . --config Release
  ```
7. 执行命令`./FASTER`,随后输入包含文件的目录路径即可运行，如要得到最后的json结果，需要修改test.cpp，然后重新编译
8. json结果格式
  ```
  {file_name1:[
      {'bbox': [x, y, w, h],
      'category_id': (int),
      'score': (float)},
      {'bbox': [x, y, w, h],
      'category_id': (int),
      'score': (float)},
      ...
  ],
   file_name2: ...
  }
  ```