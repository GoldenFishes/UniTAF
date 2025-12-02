import json
import numpy as np
import os.path as osp
from torch.utils import data
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from typing import Any, List

from .data_item import DataItem
from .dataset_config import dataset_config


class AudioFaceDataset(data.Dataset):
    """音频-面部数据集类，继承自PyTorch的Dataset类"""

    def __init__(self, data_label_path: str, args: dict = None) -> None:
        """初始化数据集

        Args:
            data_label_path: 数据标签文件的路径
            args: 参数字典，包含各种配置参数
        """
        super().__init__()  # 调用父类构造函数
        data_root = osp.dirname(data_label_path, )  # 获取数据根目录
        id_template_path = osp.join(data_root, 'id_template.npy')  # 构建身份模板文件路径
        id_template_list = np.load(id_template_path)  # 加载身份模板数据

        with open(data_label_path) as f:
            labels = json.load(f)  # 加载标签JSON文件

        info = labels['info']  # 获取信息部分
        self.id_list = info['id_list']  # 身份ID列表
        self.sr = 16000  # 采样率设为16kHz
        data_info_list = labels['data']  # 获取数据信息列表
        data_list = []  # 初始化数据列表

        # 遍历所有数据信息
        for data_info in tqdm(data_info_list):
            # 创建数据项对象
            data = DataItem(
                annot_path=osp.join(data_root, data_info['annot_path']),  # 注释文件路径
                audio_path=osp.join(data_root, data_info['audio_path']),  # 音频文件路径
                identity_idx=data_info['id'],  # 身份索引
                annot_type=data_info['annot_type'],  # 注释类型
                dataset_name=data_info['dataset'],  # 数据集名称
                id_template=id_template_list[data_info['id']],  # 身份模板
                fps=data_info['fps'],  # 帧率
                processor=args.processor,  # 音频处理器
            )
            if data.duration < 0.5:  # 如果音频时长小于0.5秒则跳过
                continue
            data_list.append(data)  # 将数据项添加到列表

        self.data_list = data_list  # 设置数据列表
        return

    def __len__(self, ):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:
        data = self.data_list[index]
        return data.get_dict()

    def get_identity_num(self, ):
        """获取身份数量"""
        return len(self.id_list)


class MixAudioFaceDataset(AudioFaceDataset):
    """混合音频-面部数据集，用于合并多个数据集"""

    def __init__(self,
                 dataset_list: List[AudioFaceDataset],
                 duplicate_list: list = None) -> None:
        """初始化混合数据集

        Args:
            dataset_list: 数据集列表
            duplicate_list: 数据集重复次数列表，用于数据增强
        """
        super(AudioFaceDataset).__init__()  # 调用父类构造函数
        self.id_list = []  # 初始化身份列表
        self.sr = 16000  # 采样率
        self.data_list = []  # 初始化数据列表
        self.annot_type_list = []  # 初始化注释类型列表

        # 如果没有提供重复列表，则创建默认列表
        if duplicate_list is None:
            duplicate_list = [1] * len(dataset_list)  # 每个数据集重复1次
        else:
            assert len(duplicate_list) == len(dataset_list)  # 确保长度一致

        # 遍历数据集和对应的重复次数
        for dup, dataset in zip(duplicate_list, dataset_list):
            id_index_offset = len(self.id_list)  # 计算身份索引偏移量

            # 对数据集中的每个数据项调整身份索引
            for d in dataset.data_list:
                d.offset_id(id_index_offset)  # 偏移身份ID
                # 收集注释类型
                if d.annot_type not in self.annot_type_list:
                    self.annot_type_list.append(d.annot_type)

            # 合并身份列表和数据列表
            self.id_list = self.id_list + dataset.id_list
            self.data_list = self.data_list + dataset.data_list * dup  # 根据重复次数复制数据
        return


def get_dataset_list(args):
    """获取数据集列表

    Args:
        args: 参数字典

    Returns:
        训练数据集列表
    """
    dataset_name_list = args.dataset  # 数据集名称列表
    # 获取数据集目录列表
    dataset_dir_list = [
        dataset_config[name]['dirname'] for name in dataset_name_list
    ]
    # 构建训练JSON文件路径列表
    train_json_list = [
        osp.join(args.data_root, dataset_name, args.train_json_key)
        for dataset_name in dataset_dir_list
    ]

    args.read_audio = False  # 设置不读取音频
    args.processor = None  # 初始化处理器为空

    train_dataset_list = []  # 初始化训练数据集列表

    # 加载音频特征提取器
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo)
    args.processor = processor  # 设置处理器

    # 为每个训练JSON文件创建数据集
    for train_json in train_json_list:
        dataset = AudioFaceDataset(
            train_json,
            args,
        )
        train_dataset_list.append(dataset)

    return train_dataset_list


def get_single_dataset(args, json_path: str = ''):
    """获取单个数据集

    Args:
        args: 参数字典
        json_path: JSON文件路径

    Returns:
        测试数据加载器
    """
    args.read_audio = True  # 设置读取音频
    # 加载音频特征提取器
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo)
    args.processor = processor  # 设置处理器

    # 创建数据集
    dataset = AudioFaceDataset(
        json_path,
        args,
    )
    # 创建数据加载器
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,  # 批大小为1
        shuffle=False,  # 不洗牌
        pin_memory=True,  # 使用锁页内存
        num_workers=args.workers)  # 工作进程数

    return test_loader


def get_dataloaders(args):
    """获取所有数据加载器（训练、验证、测试）

    Args:
        args: 参数字典

    Returns:
        训练、验证、测试数据加载器
    """
    dataset_name_list = args.dataset  # 数据集名称列表
    print(dataset_name_list)
    # 获取数据集目录列表
    dataset_dir_list = [
        dataset_config[name]['dirname'] for name in dataset_name_list
    ]
    # 构建训练、验证、测试JSON文件路径列表
    train_json_list = [
        osp.join(args.data_root, dataset_dir, args.train_json_key)
        for dataset_dir in dataset_dir_list
    ]
    val_json_list = [
        osp.join(args.data_root, dataset_dir, 'val.json')
        for dataset_dir in dataset_dir_list
    ]
    test_json_list = [
        osp.join(args.data_root, dataset_dir, 'test.json')
        for dataset_dir in dataset_dir_list
    ]

    # 加载音频特征提取器
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo)
    args.processor = processor  # 设置处理器

    # 训练数据加载器
    if getattr(args, 'do_train', True) is True:
        train_dataset_list = []
        for train_json in train_json_list:
            dataset = AudioFaceDataset(train_json, args)
            train_dataset_list.append(dataset)
        # 创建混合训练数据集
        mix_train_dataset = MixAudioFaceDataset(train_dataset_list,
                                                args.duplicate_list)
        train_loader = data.DataLoader(
            dataset=mix_train_dataset,
            batch_size=args.batch_size,  # 批大小
            shuffle=True,  # 洗牌
            pin_memory=True,  # 使用锁页内存
            num_workers=args.workers)  # 工作进程数
    else:
        train_loader = None

    # 验证数据加载器
    if getattr(args, 'do_validate', True) is True:
        val_dataset_list = []
        for val_json in val_json_list:
            dataset = AudioFaceDataset(val_json, args)
            val_dataset_list.append(dataset)
        # 创建混合验证数据集
        mix_val_dataset = MixAudioFaceDataset(val_dataset_list)
        val_loader = data.DataLoader(
            dataset=mix_val_dataset,
            batch_size=1,  # 批大小为1
            shuffle=False,  # 不洗牌
            pin_memory=True,  # 使用锁页内存
            num_workers=args.workers)  # 工作进程数
    else:
        val_loader = None

    # 测试数据加载器
    if getattr(args, 'do_test', True) is True:
        test_dataset_list = []
        for test_json in test_json_list:
            dataset = AudioFaceDataset(test_json, args)
            test_dataset_list.append(dataset)
        # 创建混合测试数据集
        mix_test_dataset = MixAudioFaceDataset(test_dataset_list)
        test_loader = data.DataLoader(
            dataset=mix_test_dataset,
            batch_size=1,  # 批大小为1
            shuffle=False,  # 不洗牌
            pin_memory=True,  # 使用锁页内存
            num_workers=args.workers)  # 工作进程数
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
