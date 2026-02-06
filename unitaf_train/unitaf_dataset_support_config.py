'''
UniTAF Dataset支持的数据集，

多数据集训练功能参考UniTalker实现，
- D12 从网络中获取的一般质量的视频数据，转录成a2f数据集，后经过处理为UniTAF数据集格式。
- D13 从mead情感数据集中获取的子集，经过处理为UniTAF数据集格式。主要用于测试情感控制的训练。
'''

unitaf_dataset_support_config = {
    'D12': {
        'dirname': 'D12_unitalker_0326',   # 从根目录中获取
        'annot_type': 'qxsk_inhouse_blendshape_weight',   # 趣像时空qxsk，室内数据集
        'scale': 1.0,
        'annot_dim': 61,   # arkit格式的维度
        'subjects': 20,    # 说话人数量，对应数据集.json中["info"]["id_list"]中元素数量
        'pca': False,
    },
    'D13': {
        'dirname': 'D13_emotion_unitaf_260109',   # 从根目录中获取
        'annot_type': 'qxsk_inhouse_blendshape_weight',   # 趣像时空qxsk，室内数据集
        'scale': 1.0,
        'annot_dim': 61,   # arkit格式的维度
        'subjects': 176,    # 说话人数量，对应数据集.json中["info"]["id_list"]中元素数量
        'pca': False,
    },
}