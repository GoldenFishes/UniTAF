'''
UniTAF Dataset支持的数据集，

多数据集训练功能参考UniTalker实现，
但是目前只有D12按照我们的数据集标准准备，其他的尚未支持，需要经过预处理后才可添加
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
}