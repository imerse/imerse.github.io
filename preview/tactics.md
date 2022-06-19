#### Loss 系列

``` mermaid
graph TD
    S("loss") --"交叉熵"--> CE("CE loss") --"分类不平衡/难样本挖掘"--> FL("Focal loss") 

    S --> IO("iou loss") --"+外接矩形"--> GI("Giou loss") 
    --"+中心距离/尺度"--> DI("Diou loss") 
    --"+长宽比"--> CI("Ciou loss") 

    S --"解决语义分割正负样本强烈不平衡"---> DIC("dice loss") 

    S --"序列不对齐问题"--> CT("CTC loss")

    S --"?"---> NC("NCE loss")--"?"--> INC("infoNCE loss")

    S --> L1("L1 loss") --> SL1("smoothL1 loss")
    S --> L2("L2(MSE) loss") 
```


#### 压缩 系列

- 剪枝：利用BN中的gamma作为通道的缩放因子，因为gamma是直接作用在特征图上的，值越小，说明该通道越不重要，可以剔除压缩模型。为了获取稀疏的gamma分布，便于裁剪。论文将L1正则化增加到gamma上。本文提出的方法简单，对网络结构没有改动，效果显著。

``` mermaid
graph TD
    S("压缩") --"量化"--> IN("int8")

    S--"BN gamma+L1正则"--> SL("通道剪枝")
    S--"softmax-T;平滑标签"--> KD("知识蒸馏")



```



#### 可视化 系列

- CAM: 利用全局平均池化获取特征向量，再和输出层进行全连接

``` mermaid
graph TD
    S("特征可视化") --"量化"--> CA("CAM")
    --"-GAP层"--> GR("grad cam")



```

#### nms 系列


``` mermaid
graph TD
    S("nms") --"抑制,线性或高斯"--> SO("Soft-NMS")
    --"?"--> SOR("Softer-NMS")



```