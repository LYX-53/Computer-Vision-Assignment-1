import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
from ultralytics import YOLO  # 导入YOLO模型
import torch.serialization#
from torch.nn.modules.container import Sequential
from ultralytics.nn.tasks import SegmentationModel, DetectionModel
from ultralytics.nn.modules import Conv  # 新增
from QtFusion.path import abs_path
import matplotlib
matplotlib.use('TkAgg')
# 全面安全类注册，我的pytorch=2.6出现的问题，查deepseek之后修改了，禁用AMP
torch.serialization.add_safe_globals([
    torch.nn.Module,
    SegmentationModel,
    DetectionModel,
    Conv,  # 新增关键类
    Sequential,
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    # 添加其他可能用到的类
    *[cls for name, cls in vars(torch.nn).items()
      if isinstance(cls, type) and issubclass(cls, torch.nn.Module)],
])

# 禁用AMP检查
os.environ['YOLO_AMP'] = '0'

if __name__ == '__main__':  # 确保该模块被直接运行时才执行以下代码
    workers = 1
    batch = 2  # 适当等修改Batchsize，根据电脑等显存/内存设置，如果爆显存可以调低(8-32,最低2),这里我用8爆显存了，改为用2
    # 设置设备为 GPU（如果可用）或者 CPU，这里我使用gpu训练，用cpu试了一下，训练1/100要30多分钟，改为使用gpu之后训练1/100只需要1分几秒；
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 打印出正在使用的设备
    print(f"Using device: {device}")  # 新增的部分
    data_path = abs_path(f'datasets/data/data.yaml', path_type='current')  # 数据集的yaml的绝对路径

    unix_style_path = data_path.replace(os.sep, '/')
    # 获取目录路径
    directory_path = os.path.dirname(unix_style_path)
    # 读取YAML文件，保持原有顺序
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # 修改path项
    if 'train' in data and 'val' in data and 'test' in data:
        data['train'] = directory_path + '/train'
        data['val'] = directory_path + '/val'
        data['test'] = directory_path + '/test'

        # 将修改后的数据写回YAML文件
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)
    # 不同模型大小不同，对设备等要求不同，如果要求较高的模型【报错】则换其他模型测试即可（例如yolov8-seg.yaml、yolov8-seg-goldyolo.yaml、yolov8-seg-C2f-Faster.yaml、yolov8-seg-C2f-DiverseBranchBlock.yaml、yolov8-seg-efficientViT.yaml等）
    #这里我使用yolov8-seg-efficientViT.yaml速度快一点
    model = YOLO(r"D:\桌面\计算机视觉第一次作业（户型图分割）\House_cut_apart_code\改进的YOLOv8模型的配置文件\yolov8-seg-efficientViT.yaml")
    #model = YOLO(model='./yolov8s-seg.yaml', task='segment').load('./weights/yolov8s-seg.pt')  # 加载预训练的YOLOv8模型
    ckpt = torch.load('./weights/yolov8s-seg.pt', map_location='cpu', weights_only=False)

    # 转换参数名格式，pytorch=2.6出现的问题
    state_dict = {}
    for k, v in ckpt['model'].named_parameters():
        new_key = k.replace('model.', '')  # 去掉前缀
        state_dict[new_key] = v

    model.model.load_state_dict(state_dict, strict=False)

    results = model.train(  # 开始训练模型
        data=data_path,  # 指定训练数据的配置文件路径
        device=device,  # 自动选择进行训练
        workers=workers,  # 指定使用2个工作进程加载数据
        imgsz=640,  # 指定输入图像的大小为640x640
        epochs=100,  # 指定训练100个epoch
        batch=batch,  # 指定每个批次的大小为8
        amp=False,  # 禁用AMP
        exist_ok=True

    )
