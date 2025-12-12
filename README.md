# 机器学习课程项目&作业（2025 Fall, THU）

## Project 1: 从零实现KNN和SVM

### 简介
本项目从零实现了KNN和SVM两种算法。1）我们实现了基于L1和L2距离的KNN算法，在自建数据集上测试了算法性能，分析了超参数对于算法表现的影响，绘制了决策边界，
在HIGGS数据集上测试了两种算法的准确率。2）我们实现了基于梯度下降的线性SVM、基于随机梯度下降的线性SVM和基于script.minmize()方法的RBF核SVM。在自制数据
集上分析了算法的性能核超参数的影响，绘制了决策边界，在HIGGS数据集上测试了算法的准确率。此外，我们还利用sklearn机器学习库自动实现了一个RBF核SVM，测试了大样本下基于核方法的SVM的分类能力。

项目报告在~/project1/report.

### 快速开始

Step1: 下载project1文件夹并进入，修改.sh文件中的路径。

Step2: 创建环境并安装所需要的库

```
conda create -n ml_hw1 python==3.10
conda activate ml_hw1
pip install -r requirements.txt
```

Step3: 运行生成数据集代码
```
bash run_generate.sh
```
注意：HIGGS数据集在 https://pan.quark.cn/s/ed1b7a009fa3. 请在~/project1/datasets文件夹下创建HIGGS文件夹，并将下载的HIGGS.csv文件放入其中。

Step4: 依次运行bash文件。结果将打印在终端或保存在results文件夹中。我们建议在运行run_SVM-HIGGS.sh时注释掉不需要的代码，只保留需要运行的部分。因为SVM模型在HIGGS数据集上训练耗时较长，依次运行便于监控训练过程和及时调试。

## Project 2: 在OGBN数据集上实现MLP,GCN,LPA和LLM

### 简介
本项目基于OGBN的arxiv、proteins和product数据集实现并测试了四种基线算法：MLP、LPA、GCN和大语言模型（LLM）。

项目报告在~/project2/report.

### 快速开始

Step1: 下载project2文件夹并进入.

Step2: 创建环境并安装所需要的库

```
conda create -n ml_hw2 python==3.10
conda activate ml_hw2
pip install -r requirements.txt
```

Step3: 处理所需要的数据集
```
load_ogb.py:下载ogb数据集
load_arxiv.py:下载ogb_arxiv的文章标题和摘要
pre_see:打印.gz文件中的前若干行进行预览
create_proteins_feature.py:为ogbn-proteins的每个节点制作特征（入边的平均值）
split_gcn_lpa.py:为GCN和LPA创建所需要的数据集
```

Step4: 依次运行models里面的模型代码。结果将会保存在同级目录下的./result中
