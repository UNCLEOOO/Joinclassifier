# JoinClassifier

该项目用于在数据湖中发现可以连接的表对。

## Dataset
本项目采用 PolyJoin [1] 的数据集构造方式进行训练集以及测试集的构造。

PolyJoin 是一种用于在数据湖中发现多键语义可连接表的方法，它通过多键编码器和自监督学习技术来识别可以进行语义连接的表对。本项目借鉴了其数据集构造方法，用于训练和评估表连接分类器。

### 引用文献
[1] Xuming Hu et al. **PolyJoin: Semantic Multi-key Joinable Table Search in Data Lakes**. In *Findings of the Association for Computational Linguistics: NAACL 2025*.

## 性能评估

### JoinClassifier 效果
|P@30|R@30|MAP@30|
|----|----|------|
|0.7386|0.6331|0.8551|

### PolyJoin 基准效果
|P@30|R@30|MAP@30|
|----|----|------|
|0.6402|0.5488|0.6928|

### 性能对比
与 PolyJoin 基准方法相比，JoinClassifier 在所有指标上都取得了显著提升：
- **P@30**: 提升 15.4% (0.7386 vs 0.6402)
- **R@30**: 提升 15.4% (0.6331 vs 0.5488)
- **MAP@30**: 提升 23.4% (0.8551 vs 0.6928)

这些结果表明 JoinClassifier 在数据湖中发现可连接表对的任务上具有更强的性能。
