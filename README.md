# Recommend-System-in-China-Telecom

基于PyTorch的双塔深度学习推荐模型，实现用户-商品匹配预测

![双塔模型结构图]![image](https://github.com/user-attachments/assets/88acd22c-9bb9-4fb4-913a-abe30f22fc35)


## 项目特点

- 🏗️ **双塔神经网络架构**（用户塔 + 商品塔）
- 🔢 **多类型特征处理**：支持数值型和类别型特征
- 🧠 **深度ID嵌入网络**：使用MLP动态生成用户/商品嵌入
- ⚖️ **类别平衡处理**：通过负样本上采样解决数据不平衡问题
- ⚡ **多硬件加速**：自动检测CUDA/MPS/CPU设备
- 📊 **可视化监控**：集成TensorBoard训练指标追踪

## 项目结构
