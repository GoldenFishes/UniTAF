dataset_config = {
    'D0': {
        'dirname': 'D0_BIWI',
        'annot_type': 'BIWI_23370_vertices',
        'scale': 0.2,
        'annot_dim': 23370 * 3,
        'subjects': 6,
        'pca': True,
    },
    'D1': {
        'dirname': 'D1_vocaset',
        'annot_type': 'FLAME_5023_vertices',
        'scale': 1.0,
        'annot_dim': 5023 * 3,
        'subjects': 12,
        'pca': True,
    },
    'D2': {
        'dirname': 'D2_meshtalk',
        'annot_type': 'meshtalk_6172_vertices',
        'scale': 0.001,
        'annot_dim': 6172 * 3,
        'subjects': 13,
        'pca': True,
    },
    'D3': {
        'dirname': 'D3D4_3DETF/D3_HDTF',
        'annot_type': '3DETF_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 52,
        'subjects': 141,
        'pca': False,
    },
    'D4': {
        'dirname': 'D3D4_3DETF/D4_RAVDESS',
        'annot_type': '3DETF_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 52,
        'subjects': 24,
        'pca': False,
    },
    'D5': {
        'dirname': 'D5_unitalker_faceforensics++',
        'annot_type': 'flame_params_from_dadhead',
        'scale': 1.0,
        'annot_dim': 413,
        'subjects': 719,
        'pca': False,
    },
    'D6': {
        'dirname': 'D6_unitalker_Chinese_speech',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 8,
        'pca': False,
    },
    'D7': {
        'dirname': 'D7_unitalker_song',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 11,
        'pca': False,
    },
    'D12': {
        'dirname': 'D12_UniTextAudioFace',   # 从根目录中获取
        'annot_type': 'qxsk_inhouse_blendshape_weight',   # 趣像时空，室内数据集
        'scale': 1.0,
        'annot_dim': 61,   # arkit格式的维度
        'subjects': 20,    # 说话人数量，对应数据集.json中["info"]["id_list"]中元素数量
        'pca': False,
    },
}
