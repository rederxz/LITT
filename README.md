### 实验记录表

- google sheet: https://docs.google.com/spreadsheets/d/1tITOX-v9k_mdmE0t48XfDtSBot3XKt-gMJbbh1zz1v0/edit#gid=90347781
- train script tab: 记录了训练实验
  - res：图像分辨率
  - mask：欠采样方式 CS or kt-BLAST
  - rate：加速倍数，保留k空间中心的条数
  - online mask：欠采样掩膜是否在训练过程中随机生成
  - bs：BATCH SIZE
  - optimizer：优化器及参数
  - loss：损失函数
  - base_lr：初始学习率大小
  - epochs：迭代轮次
  - lr_schedule：学习率衰减方式
  - wd：weight decay权重衰减
  - grad clip：梯度裁剪的范围，若为空则没有使用
  - train loss：训练损失
  - test loss：测试损失
  - PSNR：
  - Mag SSIM：
  - Phase RMSE：这里计算了全图的相位RMSE，仅作参考
  - comment：其他备注
  - script：对应的训练脚本名，脚本文件在./train_scripts文件夹中
- inference script tab: 记录了测试实验
  - model arch：模型架构
  - model source：模型权重通过哪一个训练脚本训练得到
  - img source：图像来源
    - patient25：LITT数据集中的25号患者
    - test_data xx：实采数据的图像，具体详情可见 test data tab中
  - mask source：
    - cs_8x_c8这种格式的：指预先生成的欠采样掩膜
    - test_data xx：实采数据的掩膜
  - comment：备注
  - script：对应的测试脚本名，脚本文件在./test_scripts文件夹中
- test data tab: 记录了测试实验用到的实采数据
- old train script tab：老训练脚本的记录，已废弃

### 代码

- github代码仓库（experimental分支）：https://github.com/rederxz/LITT/tree/experimental

- 结构：

  ```bash
  ├─ds_split # 数据划分文件夹
  │  ├─test
  │  ├─train
  │  └─val
  ├─inference_scripts # 测试脚本文件夹
  ├─model # 模型文件夹，其中的py文件定义了模型结构
  ├─sub_ds_split # 数据（子集，10位患者）划分文件夹
  │  ├─test
  │  ├─train
  │  └─val
  ├─train_scripts # 训练脚本文件夹
  ```

- 运行方法：

  ```bash
  cd path-to-code-folder/train_scripts # 或者 inference_scripts
  python 脚本名.py
  ```