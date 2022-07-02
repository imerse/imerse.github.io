

#### 目标检测 系列
``` mermaid
graph TD
    FR(Faster rcnn) --"cascade回归,逐stage提高proposal的IoU值"--> cascadeRCNN(cascadeRCNN)

    subgraph 单阶段
    SSD
    YOLOv1("YOLOv1, M*N，无anchor") --"anchor(kmeans, wh), iou匹配"--> YOLOv2 --"多尺度预测"--> YOLOv3 -- "加权残差WRC,CSP, CmBN, SAT,Mish,Mosaic,CIou" --> YOLOv4(YOLOv4) --"改进匹配规则,backbone, fpn+pan"--> YOLOv5
    YOLOv4 --> s("scaled YOLOv4")
    YOLOv5 --"根据当前帧预测下一帧中目标位置"-->StreamYolo
    YOLOv5 --"无anchor,"--> YOLOX?
    end
    
    FPN --> YOLOv5
    
    subgraph anchor free
    anchorfree --"bottom-up,关键点-匹配,embedding vector"--> CornerNet --"热图预测中心点,add wh/offset分支"--> CenterNet -- "FPN,centerness过滤低质量样本,add 正样本召回" --> FCOS巅峰
    end
```

#### 级联目标检测 系列
``` mermaid
graph TD
    %% this is a comment A -- text --> B{node}
    cascadeRCNN(cascadeRCNN,cascade回归,逐stage提高proposal的IoU值)
```


#### 分割 系列

- 先占坑


``` mermaid
graph TD
    %% segmentation
    
```


#### transformer 系列

- DETR[202005]：端到端的目标检测，no anchor/nms，提出了一个新的基于集合的目标函数；限制输出100个框，直接限制阈值；【主要还是 transformer 的全局特征提的很好】
  - 小目标效果不好
  - 500epoch
  - 采用object query来替代了anchor机制
  - 利用二分图匹配来替代nms

``` mermaid
graph TD
    TR(Transformer) --"输入一对图片，遥感影像"--> ChangeFormer
    TR --"细粒度分类,PSM,contrastive loss"--> TransFG
    TR --"应用于Vision"--> VI(VIT)--"滑窗,窗口内SA,多尺度,SWA,Att Mask"--> ST("swin Transformer")

    TR --"-nms,全局建模,no proposal"--> DE(DETR) 
    --"deformable"--> DDE("deformable DETR")

```


#### 人脸 系列
``` mermaid
graph TD
    S("人脸检测") --"add 5landmarks(multi-task),context modeling,dcn,light weight"--> RE(retinaface1905) --"nas,计算和样本搜索分配"--> SC("SCRFD2105")
    
```
---
``` mermaid
graph TD
    S("人脸属性") --> FA(FairFace) 
    
```

#### OCR 系列
``` mermaid
graph TD
    SJC("OCR-检测") --"可微分thresholdmap"--> DB(DBnet) 
    SSB("OCR-识别") --"CNN+RNN+CTC loss"--> CR(CRNN) 

    DB-->CR

    SJC--> EA(EAST)
    EA --先检测--> BJ(文本编辑) --"文本骨架,文字移除,合成"--> SRNet
```

#### 对比学习 系列

- 无监督学习，训练越久模型越大一般效果也确实越好。
- 负样本一定要多

趋势：

- 目标函数：infoNCE或者相关变体
- 模型：一个encoder + mlp prjection head
- 数据增强：更强大的aug
- 动量编码器
- 训练时间更长, 更大的batch size

`multi-crop:` 改变只有1+1两个正样本对的情况，多加一些小尺寸Crop，在保证计算量不大幅增加的同时，增加全局和局部的view，有效提点。

``` mermaid
graph TD
    S("对比学习cv双雄") --"字典查询,1队列2动量编码器"--> MO(MOCO) -- "+mlp,aug,cos,+epochs"--> MOCOV2  -- "freeze tokenization"--> MOCOV3 
    S --"resnet50"--> SimCLR --"resnet152,SK(selective kernel),+1层mlp,+动量编码器"--> SimCLRV2

    S --"swap prediction,3000聚类中心,multi-crop"--> SW(SwAV)


    S1("对比学习-无负样本") --"无负样本,用正样本1的特征预测正样本2"--> BY(BYOL) --"-BN,gn+ws"--> BY2(BYOL2)
    S1 --"-batchsize,-动量,-负样本;EM"--> Sim("SimSiam") 

    Sim ----> MOCOV3
    SimCLR --> BY

```



一张图总结所有。

![20220612161231](https://lcv1-1256975222.cos.ap-shanghai.myqcloud.com/20220612161231.png)


#### GAN 系列

``` mermaid
graph TD
    S("GAN系列") --"G+D"--> GAN --"加入condition控制类别" --> CG(CGAN)
    --"+CNN(BN+Relu..)"-->DC(DCGAN)
    CG --> pixpix --循环一致loss--> CycleGAN

    GAN --"Wasserstein距离解决梯度消失问题"--> W-GAN
```


