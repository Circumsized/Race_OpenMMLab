'''
定义了一个名为_base_的list，list中包含三个路径，分别指向三个.py文件
'''
_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py','../_base_/default_runtime.py']

'''
创建一个名为model的dict，其中
head(头部)包含了num_classes参数和topk参数
'''
model = dict(
    head=dict(
        #num_classes参数指定了训练的类别数量
        num_classes=5,
        #topk参数指定了要预测的top k个类别
        topk = (1,)
    ))

#创建了一个名为data的dict，其中包含了 4个dict参数
data = dict(
    #指定训练所用样本数量
    samples_per_gpu = 32,
    #指定每个gpu使用worker数量
    workers_per_gpu = 2,

   #指定训练数据集前缀，注解文件和类标文件
    train = dict(
        data_prefix = 'data/flower/train',
        ann_file = 'data/flower/train.txt',
        classes = 'data/flower/classes.txt'
    ),

    #指定测试数据集前缀，注解文件和类标文件
    val = dict(
    data_prefix = 'data/flower/val',
    ann_file = 'data/flower/val.txt',
    classes = 'data/flower/classes.txt'
    )
)

'''
创建了一个名为optimizer的dict，其中包含type, lr, momentum, weight_decay四个参数，
type参数指定了优化器的类型，lr参数指定了学习率，
momentum参数指定了动量参数，weight_decay参数指定了L2正则化系数
'''
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

'''
创建了一个名为optimizer_config字典，
其中包含grad_clip参数，grad_clip参数用于控制梯度更新时的最大范数
'''
optimizer_config = dict(grad_clip=None)

#创建了一个名为runner的dict，其中包含type和max_epochs两个参数
lr_config = dict(
    policy='step',
    step=[1])
#ype参数指定了运行器的类型，max_epochs参数指定了训练最大的epoch次数
runner = dict(type='EpochBasedRunner', max_epochs=100)

 # 预训练模型,指定了预训练的模型文件的路径。
load_from ='/HOME/scz0auh/run/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'