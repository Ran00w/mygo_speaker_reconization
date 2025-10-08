# BanG Dream! It's MyGO!!!!! 语音识别系统

![Banner](/static/images/mygo.jpg)

## 🚀 项目介绍

本系统是基于Transformer架构的《BanG Dream! It's MyGO!!!!!》角色语音识别解决方案，整合了以下核心技术：

🔥 **核心架构**
- 2层Transformer Encoder堆叠结构
- 40维MFCC语音特征提取
- 语音活动检测(VAD)算法
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
- 🎤 支持MyGO!!!!!全员语音识别（Anon、Rana、Soyo、Taki、Tomori）
- 🔥 高准确率的测试结果
- ⚡ 实时推理响应
- 🌐 直观的Web界面

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
```

## 📂 项目结构
```bash
├── 📁 models/                   # 模型相关
│   ├── __init__.py
│   └── classifier.py            # 分类器模型
├── 📁 utils/                    # 工具函数
│   ├── __init__.py
│   ├── audio_utils.py           # 音频处理工具
│   ├── feature_extraction.py    # 特征提取工具
│   └── data_utils.py            # 数据处理工具
├── 📁 datasets/                 # 数据集相关
│   ├── __init__.py
│   └── dataset.py               # 数据集类
├── 📁 scripts/                  # 脚本文件
│   ├── train.py                 # 训练脚本
│   ├── test.py                  # 测试脚本
│   └── preprocess.py            # 数据预处理脚本
├── 📁 static/                   # 静态资源
│   ├── css/                     # CSS文件
│   ├── js/                      # JavaScript文件
│   └── images/                  # 图片资源
├── 📁 templates/                # HTML模板
│   └── index.html               # 主页面
├── 📁 data/                     # 数据目录
│   ├── train.tsv                # 训练数据列表
│   ├── dev.tsv                  # 验证数据列表
│   ├── test.tsv                 # 测试数据列表
│   ├── Anon/                    # Anon角色音频
│   ├── Rana/                    # Rana角色音频
│   ├── Soyo/                    # Soyo角色音频
│   ├── Taki/                    # Taki角色音频
│   └── Tomori/                  # Tomori角色音频
├── 📁 checkpoints/              # 模型检查点
│   └── model.ckpt               # 训练好的模型
├── 📁 uploads/                  # 上传文件目录
├── 📁 segments/                 # 音频分割片段目录
├── 📄 app.py                    # Flask应用入口
├── 📄 config.py                 # 配置文件
├── 📄 requirements.txt          # 依赖包列表
└── 📄 README.md                 # 项目说明文档
```

## 🚀 使用方法

1. 在网页界面点击"选择文件"按钮，上传音频或视频文件
2. 点击"上传并识别"按钮，系统将自动处理文件并识别说话人
3. 查看识别结果，系统会以不同颜色高亮显示当前说话的角色
4. 可以播放上传的音频或视频文件进行验证

## 📚 训练自定义模型

如果需要使用自己的数据集训练模型，请按照以下步骤操作：

1. 将音频文件按照说话人分类存放在不同的文件夹中
2. 修改`utils/data_utils.py`中的`split_data2()`函数，更新说话人名称和对应标签
3. 运行数据预处理脚本：
   ```bash
   python scripts/preprocess.py --split_data
   ```
4. 运行训练脚本：
   ```bash
   python scripts/train.py --batch_size 32 --steps 5000 --learning_rate 1e-3
   ```

## 📊 测试模型

使用测试集评估模型性能：

```bash
python scripts/test.py --save_path checkpoints/model.ckpt
```

## ⚙️ 参数说明

### 训练参数

- `--batch_size`: 批次大小，默认为32
- `--save_path`: 模型保存路径，默认为"checkpoints/model.ckpt"
- `--steps`: 训练步数，默认为5000
- `--n_spk`: 说话人数量，默认为5
- `--warmup_steps`: 预热步数，默认为100
- `--learning_rate`: 学习率，默认为1e-3
- `--valid_steps`: 验证步数，默认为50

### 数据预处理参数

- `--split_data`: 是否分割数据集

## 🔧 技术细节

### 特征提取

系统使用MFCC（Mel频率倒谱系数）作为音频特征，具体步骤如下：
1. 读取音频文件
2. 如果是双声道音频，转换为单声道
3. 计算帧长和NFFT
4. 提取40维MFCC特征
5. 对特征进行截取或填充，统一长度为96帧

### 模型架构

使用基于Transformer的分类器模型，主要组件包括：
1. 预网络：将40维MFCC特征映射到80维
2. Transformer编码器：2层Transformer编码器层，每层包含2个注意力头
3. 预测层：将编码器输出映射到说话人类别

### 后处理

模型输出经过后处理提高稳定性：
1. 使用滑动窗口平滑预测结果
2. 过滤短时切换（小于4帧的切换被视为噪声）

## ❓ 常见问题

1. **Q: 上传文件后提示"File type not allowed"**
   A: 请确保上传的文件格式为wav、mp3、mp4、avi或mkv


2. **Q: 训练过程中出现CUDA内存不足**
   A: 减小batch_size参数，例如：`python scripts/train.py --batch_size 16`

## 📝 数据来源

数据训练集可从B站up主[椎名乐奈](https://space.bilibili.com/320151977?spm_id_from=333.337.search-card.all.click)处获得，将各个角色的语音包放入raw_data文件夹然后对每个角色分别单独建一个文件夹，然后放入对应角色的语音音频，然后再通过split_wav.py分割音频到data文件夹中


## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

感谢BanG Dream! It's MyGO!!!!!的制作团队提供了优秀的角色语音素材。

