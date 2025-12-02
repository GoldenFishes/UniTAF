import librosa
import numpy as np
from typing import Any

from .dataset_config import dataset_config


class DataItem:
    """数据项类，用于处理单个音频-面部数据样本"""

    def __init__(self,
                 annot_path: str,
                 audio_path: str,
                 identity_idx: int,
                 annot_type: str = '',
                 dataset_name: str = '',
                 id_template: np.ndarray = None,
                 fps: float = 30,
                 processor=None) -> None:
        """初始化数据项

        Args:
            annot_path: 注释文件路径（面部运动数据）
            audio_path: 音频文件路径
            identity_idx: 身份索引
            annot_type: 注释类型
            dataset_name: 数据集名称
            id_template: 身份模板数据
            fps: 帧率（帧每秒）
            processor: 音频处理器
        """
        self.annot_path = annot_path  # 注释文件路径
        self.audio_path = audio_path  # 音频文件路径
        self.identity_idx = identity_idx  # 身份索引
        self.fps = fps  # 帧率

        # 加载注释数据（面部运动数据）
        self.annot_data = np.load(annot_path)

        # 帧率标准化处理
        if self.fps == 60:
            # 60fps降采样到30fps，每隔一帧取一帧
            self.annot_data = self.annot_data[::2]
            self.fps = 30
        elif self.fps == 100:
            # 100fps降采样到25fps，每隔三帧取一帧
            self.annot_data = self.annot_data[::4]
            self.fps = 25
        elif self.fps != 30 and self.fps != 25:
            # 检查帧率是否支持
            raise ValueError('wrong fps')

        # 从数据集配置获取缩放比例
        scale = dataset_config[dataset_name]['scale']
        # 对注释数据进行缩放和偏移处理
        self.annot_data = self.scale_and_offset(self.annot_data, scale, 0.0)

        # 加载音频数据
        self.audio_data, sr = self.load_audio(audio_path)

        # 截断音频和注释数据，使其长度一致
        self.original_audio_data, self.annot_data = self.truncate(
            self.audio_data, sr, self.annot_data, self.fps)

        # 使用处理器提取音频特征
        self.audio_data = np.squeeze(
            processor(self.original_audio_data, sampling_rate=sr).input_values)
        self.audio_data = self.audio_data.astype(np.float32)  # 转换为float32

        self.annot_type = annot_type  # 注释类型

        # 对身份模板进行相同的缩放和偏移处理
        self.id_template = self.scale_and_offset(id_template, scale, 0.0)

        # 重塑注释数据形状为 (帧数, 特征数)
        self.annot_data = self.annot_data.reshape(len(self.annot_data),
                                                  -1).astype(np.float32)
        # 重塑身份模板形状为 (1, 特征数)
        self.id_template = self.id_template.reshape(1, -1).astype(np.float32)

        self.dataset_name = dataset_name  # 数据集名称
        self.data_dict = self.to_dict()  # 转换为字典格式
        return

    def load_audio(self, audio_path: str):
        """加载音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            audio_data: 音频数据数组
            sample_rate: 采样率
        """
        if audio_path.endswith('.npy'):
            # 如果是npy格式，直接加载
            return np.load(audio_path), 16000
        else:
            # 使用librosa加载音频文件，重采样到16kHz
            return librosa.load(audio_path, sr=16000)

    def truncate(self, audio_array: np.ndarray, sr: int,
                 annot_data: np.ndarray, fps: int):
        """截断音频和注释数据，使其长度一致

        Args:
            audio_array: 音频数据数组
            sr: 音频采样率
            annot_data: 注释数据数组
            fps: 注释数据帧率

        Returns:
            truncated_audio: 截断后的音频数据
            truncated_annot: 截断后的注释数据
        """
        # 计算音频时长（秒）
        audio_duration = len(audio_array) / sr
        # 计算注释数据时长（秒）
        annot_duration = len(annot_data) / fps
        # 取三者最小值作为最终时长，最大不超过20秒
        duration = min(audio_duration, annot_duration, 20)

        # 计算对应的音频样本数
        audio_length = round(duration * sr)
        # 计算对应的注释帧数
        annot_length = round(duration * fps)

        self.duration = duration  # 保存时长信息

        # 返回截断后的音频和注释数据
        return audio_array[:audio_length], annot_data[:annot_length]

    def offset_id(self, offset: int):
        """偏移身份索引

        Args:
            offset: 偏移量
        """
        self.identity_idx = self.identity_idx + offset  # 更新身份索引
        self.data_dict['id'] = self.identity_idx  # 更新字典中的身份索引
        return

    def scale_and_offset(self,
                         data: np.ndarray,
                         scale: float = 1.0,
                         offset: np.ndarray = 0.0):
        """对数据进行缩放和偏移处理

        Args:
            data: 输入数据
            scale: 缩放比例
            offset: 偏移量

        Returns:
            处理后的数据
        """
        return data * scale + offset


    def to_dict(self, ) -> Any:
        """将数据项转换为字典格式

        Returns:
            包含所有数据的字典
        """
        item = {
            'data': self.annot_data,  # 注释数据（面部运动）
            'audio': self.audio_data,  # 处理后的音频特征
            'original_audio': self.original_audio_data,  # 原始音频数据
            'fps': self.fps,  # 帧率
            'id': self.identity_idx,  # 身份索引
            'annot_path': self.annot_path,  # 注释文件路径
            'audio_path': self.audio_path,  # 音频文件路径
            'annot_type': self.annot_type,  # 注释类型
            'template': self.id_template,  # 身份模板
            'dataset_name': self.dataset_name  # 数据集名称
        }
        return item

    def get_dict(self, ):
        """获取数据字典

        Returns:
            数据字典
        """
        return self.data_dict


# 实现UniTAF数据集的DataItem类
class UniTAFDataItem:
    """
    UniTextAudioFace数据集的数据项类，用于处理单个文本-音频-表情数据样本

    相比上述UniTalker中的DataItem，我们修改了部分命名，同时我们有两个音频数据，
    - 一个是直接从数据集中提取的音频特征
    - 一个是从数据集的文本经过tts后再提取的音频特征
    """
    def __init__(
        self,
        face_path: str,
        audio_path: str,
        text_path: str,
        speaker_idx: int,
        face_type: str = "",
        face_fps: float = 30,
        text_emotion_vector: list[float] = None,
        id_template: np.ndarray = None,
        dataset_name: str = "",
        audio_processor=None,    # A2F中audio encoder
        text_processor=None,     # text2speach模型
    ) -> None:
        # (文本-音频-表情) 数据对的路径
        self.face_path = face_path
        self.audio_path = audio_path
        self.text_path = text_path

        # 说话人idx索引
        self.speaker_idx = speaker_idx
        # 表情信息，格式与帧率
        self.face_type = face_type
        self.face_fps = face_fps
        # 文本情感控制，影响文本->音频的过程
        self.text_emotion_vector = text_emotion_vector

        # 加载面部表情数据
        self.face_data = np.load(self.face_path)
        # 表情帧率标准化处理
        if self.face_fps == 60:
            # 60fps降采样到30fps，每隔一帧取一帧
            self.face_data = self.face_data[::2]
            self.face_fps = 30
        elif self.face_fps == 100:
            # 100fps降采样到25fps，每隔三帧取一帧
            self.face_data = self.face_data[::4]
            self.face_fps = 25
        elif self.face_fps != 30 and self.face_fps != 25:
            # 检查帧率是否支持
            raise ValueError('wrong fps')

        # 从数据集配置获取缩放比例
        scale = dataset_config[dataset_name]['scale']
        # 对表情数据进行缩放和偏移处理
        self.face_data = self.scale_and_offset(self.face_data, scale, 0.0)

        if self.audio_path:  # 处理数据集中的音频
            # 直接从数据集中加载音频数据
            self.dataset_audio_data, sr = self.load_audio(self.audio_path)
            # 截断音频和表情数据，使其长度一致（以两者中最小长度为准，严格确保每一帧一一对应）
            self.original_dataset_audio_data, self.face_data = self.truncate(
                self.dataset_audio_data, sr, self.face_data, self.face_fps)

            # 使用音频处理器提取音频特征
            self.dataset_audio_data = np.squeeze(
                audio_processor(self.original_dataset_audio_data, sampling_rate=sr).input_values)
            self.dataset_audio_data = self.dataset_audio_data.astype(np.float32)  # 转换为float32

        if self.text_path:
            # 从文本中使用tts推理出音频，与直接从数据集中获取的音频区别开
            text = self.load_text(self.text_path)
            # 使用传入的text_processor( tts模型 ) 进行情感控制生成 TODO：实现外部传入的这个text_processor
            self.tts_audio_data, sr = text_processor(text, self.text_emotion_vector)

            # 生成出来的tts音频会出现和表情数据严重的长度不对齐的问题
            # FIXME：这里不使用截断，而是将tts_audio_data长度缩放到face_data上，tts输出的音频向表情对齐
            #  （尽管只能在长度上对齐，但是无法在语义密度上对齐，即相同长度下每个词的分布也是不均衡的）
            self.original_tts_audio_data, self.face_data = self.truncate(
                self.tts_audio_data, sr, self.face_data, self.face_fps)

            # 使用音频处理器提取音频特征
            self.tts_audio_data = np.squeeze(
                audio_processor(self.original_tts_audio_data, sampling_rate=sr).input_values)
            self.tts_audio_data = self.tts_audio_data.astype(np.float32)  # 转换为float32


        # 对身份模板进行相同的缩放和偏移处理
        self.id_template = self.scale_and_offset(id_template, scale, 0.0)
        # 重塑表情数据形状为 (帧数, 特征数)
        self.face_data = self.face_data.reshape(len(self.face_data), -1).astype(np.float32)
        # 重塑身份模板形状为 (1, 特征数)
        self.id_template = self.id_template.reshape(1, -1).astype(np.float32)

        self.dataset_name = dataset_name  # 数据集名称
        self.data_dict = self.to_dict()  # 转换为字典格式
        return

    def load_audio(self, audio_path: str):
        """
        Returns:
            audio_data: 音频数据数组
            sample_rate: 采样率
        """
        if audio_path.endswith('.npy'):
            # 如果是npy格式，直接加载
            return np.load(audio_path), 16000
        else:
            # 使用librosa加载音频文件，重采样到16kHz
            return librosa.load(audio_path, sr=16000)

    def load_text(self, text_path: str):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            print(f"读取文本文件失败 {text_path}: {e}")
            return ""

    def truncate(self, audio_array: np.ndarray, sr: int,
                 face_data: np.ndarray, fps: int):
        """截断音频和注释数据，使其长度一致
        Returns:
            truncated_audio: 截断后的音频数据
            truncated_annot: 截断后的表情数据
        """
        # 计算音频时长（秒）
        audio_duration = len(audio_array) / sr
        # 计算注释数据时长（秒）
        face_duration = len(face_data) / fps
        # 取三者最小值作为最终时长，最大不超过20秒
        duration = min(audio_duration, face_duration, 20)

        # 计算对应的音频样本数
        audio_length = round(duration * sr)
        # 计算对应的注释帧数
        face_length = round(duration * fps)

        self.duration = duration  # 保存时长信息

        # 返回截断后的音频和表情数据
        return audio_array[:audio_length], face_data[:face_length]

    def offset_id(self, offset: int):
        """偏移身份索引
        offset: 偏移量
        """
        self.identity_idx = self.identity_idx + offset  # 更新身份索引
        self.data_dict['id'] = self.identity_idx  # 更新字典中的身份索引
        return

    def scale_and_offset(self,
                         data: np.ndarray,
                         scale: float = 1.0,
                         offset: np.ndarray = 0.0):
        """对数据进行缩放和偏移处理
        """
        return data * scale + offset

    def to_dict(self, ) -> Any:
        """将数据项转换为字典格式

        Returns:
            包含所有数据的字典
        """
        item = {
            'face_data': self.face_data,      # 已经提取的表情数据
            'dataset_audio_data': self.dataset_audio_data,                     # 处理后的数据集音频特征
            'original_dataset_audio_data': self.original_dataset_audio_data,   # 原始的数据集音频数据
            'tts_audio_data': self.tts_audio_data,                             # 处理后的tts音频特征
            'original_tts_audio_data': self.original_tts_audio_data,           # 原始的tts音频数据
            'face_fps': self.face_fps,        # 表情帧率
            'id': self.identity_idx,          # 身份索引
            'text_path': self.text_path,      # 文本路径
            'face_path': self.face_path,      # 表情路径
            'audio_path': self.audio_path,    # 音频路径
            'face_type': self.face_type,      # 表情文件类型
            'template': self.id_template,     # 身份模板
            'dataset_name': self.dataset_name # 数据集名称
        }
        return item

    def get_dict(self, ):
        """获取数据字典

        Returns:
            数据字典
        """
        return self.data_dict









