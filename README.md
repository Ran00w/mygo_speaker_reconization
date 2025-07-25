# MyGO!!!!! 语音识别系统

![Banner](mygo.jpg)

## 🚀 项目介绍

本系统是基于Transformer架构的《BanG Dream! It's MyGO!!!!!》角色语音识别解决方案，整合了以下核心技术：

🔥 **核心架构**
- 12层Transformer Encoder堆叠结构
- 40维MFCC语音特征提取
- 动态时间规整(DTW)对齐算法
- 基于注意力机制的特征融合

✨ **功能亮点**
```mermaid
flowchart LR
A[音频上传] --> B[语音分割]
B --> C[MFCC特征提取]
C --> D[Transformer推理]
D --> E[角色概率分布]
E --> F[动态平滑输出]
```


## 🎥 效果演示
可见项目中的demo.mp4

## 特性
- 🎤 支持MyGO!!!!!全员语音识别
- 🔥 98%以上的测试准确率
- ⚡ 实时推理响应

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
```
<<<<<<< HEAD

## 📂 项目结构
```bash
├── 📁 data/              # 音频数据集（WAV格式）
├── 📁 model/             # 预训练模型参数
=======
数据训练集可从B站up主[椎名乐奈](https://space.bilibili.com/320151977?spm_id_from=333.337.search-card.all.click)处获得，将各个角色的语音包放入raw_data文件夹然后对每个角色分别单独建一个文件夹，然后放入对应角色的语音音频，然后再通过split_wav.py分割音频到data文件夹中
## 📂 项目结构
```bash
├── 📁 data/              # 音频数据集（WAV格式）
>>>>>>> master
├── 📁 uploads/           # 用户上传文件存储
├── 📄 app.py             # Flask REST API入口
├── 📄 model.py           # Transformer模型架构
├── 📄 utils.py           # 音频处理工具箱
├── 📄 requirements.txt   # Python依赖库清单
└── 📄 README.md          # 项目文档
```

## 🚀 性能指标

| 指标                | 数值                  |
|---------------------|-----------------------|
| 训练设备            | NVIDIA RTX 3090       |
| 训练轮次            | 50 epochs              |

