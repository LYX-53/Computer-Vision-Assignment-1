1.datasets文件夹里面为数据集：

将数据集按比例划分为训练集（70%）、验证集（20%）、测试集（10%）， 确保模型泛化能力。

2.模型选择与权重：

选择 YOLOv8-seg-efficientViT.yaml 进行户型图分割，权重为 YOLOv8s-seg.pt

3.train2(new_model).py中为训练代码，开始时使用这里使用 yolov8s-seg.yaml 模型进行训练，Batchsize=8，在cpu上训练训练 1/100 要 30 多分钟太慢且爆显存了，这里改为选择使用

yolov8-seg-efficientViT.yaml 模型，权重为 yolov8s-seg.pt ，Batchsize=2，在gpu上训练了100次一共接近三小时。

