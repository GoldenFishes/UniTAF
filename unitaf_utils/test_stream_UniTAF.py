import os
import sys
import time
import torch
from omegaconf import OmegaConf
import torchaudio
from unitaf_train.UniTAF import UniTextAudioFaceModel
from unitaf_train_component.render import render_npz_video

'''

'''

def test_unitaf_stream(unitaf):
    text = (f"清晨的邮差在山间迷了路。雾气像牛奶，淹没了所有熟悉的杉树与石阶。他推着那辆老旧的绿色自行车，在白色的混沌中，忽然看见一栋从未见过的小木屋，窗里透出温暖的、蜂蜜色的光。"
                f"他敲了门。一位白发老妇人安静地出现，仿佛已等待多年。“请进，”她说，“有您一封信。”"
                f"邮差困惑地递出邮包。老妇人却摇头，指向他鼓囊囊制服的内侧口袋。他下意识地一摸，竟真掏出一封泛黄的信，收件人写着陌生的名字，地址恰是这栋木屋。更让他心跳停止的是，寄信人的字迹，分明属于三年前去世的母亲。"
                f"雾气在那一刻开始消散。老妇人接过信，微笑道：“她迷路了这么久，终于送到了。”邮差回身，看见来路清晰如洗，而身后的木屋与灯光，已消失不见，只剩一片在晨光中摇曳的、宁静的野山菊。")

    wav_list = []
    motion_list = []
    sampling_rate = None

    start_time = time.time()
    first_chunk_time = None
    last_chunk_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for i, stream_output in enumerate(unitaf.indextts2_unitalker_stream_inference(
            spk_audio_prompt='examples/voice_zhongli.wav',
            text=text,
            emo_alpha=0.6,
            use_emo_text=True,
            emo_text=text,  # 情感控制选择从传入的情感文本中推断，不传额外用于推断的情感文本时则直接从目标文本中推断。
            interval_silence=200,  # 200 流式生成chunk之间的静音段间隔,
            verbose=False,
            max_text_tokens_per_segment=120,
            more_segment_before=0,
        )):
        '''
        {
            "sr": sr, 
            "wav": wav_chunk, 
            "fps": 25, 
            "motion": motion_vertices,  # (T, V, 3) or None
        }
        '''
        if first_chunk_time is None:
            first_chunk_time = time.time()
            print(f"[PERF] TTFB: {first_chunk_time - start_time:.3f}s")

        now = time.time()
        print(f"[PERF] chunk {i}, dt={(now - last_chunk_time) * 1000:.1f} ms")
        last_chunk_time = now
        print(torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")

        # 收集流式生成的音频和表情
        wav = stream_output["wav"]
        if isinstance(wav, list):
            wav_list.extend(wav)
        else:
            wav_list.append(wav)
        sampling_rate = stream_output['sr']
        if stream_output["motion"] is not None:
            motion_list.append(stream_output["motion"])

    end_time = time.time()

    # # 插入静音并合并所有音频段
    # wav_list = unitaf.tts_model.insert_interval_silence(wav_list, sampling_rate=sampling_rate, interval_silence=200)  # 与 infer_generator的参数一致
    wav = torch.cat(wav_list, dim=1)  # 沿时间维度拼接
    audio_length = wav.shape[-1] / sampling_rate
    print("============== PERF SUMMARY ==============")
    print(f"TTFB: {first_chunk_time - start_time:.3f}s")
    print(f"Total inference time: {end_time - start_time:.2f}s")
    print(f"Audio length: {audio_length:.2f}s")
    print(f"RTF: {(end_time - start_time) / audio_length:.3f}")
    print("==========================================")

    wav.cpu()
    # 合并表情顶点
    motion = torch.cat(motion_list, dim=0)  # 沿时间维度拼接


    # 渲染
    wav_path = "outputs/UniTAF_stream_output.wav"
    if os.path.isfile(wav_path):
        os.remove(wav_path)
    if os.path.dirname(wav_path) != "":
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    torchaudio.save(wav_path, wav.type(torch.int16), sampling_rate)  # 保存为16位PCM
    print(">> wav file saved to:", wav_path)

    if isinstance(motion, torch.Tensor):
        if motion.is_cuda:
            motion = motion.cpu()
        else:
            motion = motion
        motion_np = motion.detach().numpy()  # 分离计算图并转换为 numpy
    else:
        motion_np = motion

    # 进行渲染
    render_npz_video(
        out_np=motion_np,
        audio_path=wav_path,  # 原始 wav 完整路径；None=无声
        out_dir="outputs",  # 想把视频/图片保存文件夹
        annot_type="qxsk_inhouse_blendshape_weight",  # 你当时传给 get_vertices 的同一字符串
        save_images=False,  # False=直接出 mp4；True=出逐帧 png
        device="cuda"  # 或 "cpu"
    )

    print(">>> UniTAF stream inference done")


if __name__ == "__main__":
    '''
    测试联合模型的流式推理 python -m unitaf_utils.test_stream_UniTAF
    '''
    # 限制可见cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    # 初始化unitaf
    train_config = {
        # 模型类型，这里用于指导训练器类训练哪些模型
        "tts_model": ["IndexTTS2"],
        "a2f_model": ["UniTalker"],
        # 模型配置
        "IndexTTS2": {
            # TTS Loss计算时设置
            "use_duration_control": False,
            "duration_dropout": 0.3,
            "text_loss_weight": 0.2,
            "mel_loss_weight": 0.8,
        },
        "UniTalker": {
            # UniTalker Decoder配置, 参数与UniTalker项目的config/unitalker.yaml一致
            "interpolate_pos": 1,
            "decoder_dimension": 256,
            "decoder_type": "conv",
            "period": 30,
            "headlayer": 1,
            # UniTalker Network
            "use_pca": True,
            "pca_dim": 256,
            # A2F Loss计算时设置
            "pca_weight": 0.01,
            # 以下需要从外部获得并更新：
            "audio_encoder_feature_dim": 768,  # 原始UniTalker Decoder接收特征维度是768，我们需要经过projector使得音频特征输出相同维度
            "identity_num": 20,  # 假设是20，需要根据不同数据集决定
        },
        # 数据集类
        "dataset_config": {
            "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",
            "dataset_list": ["D12"]  # 这里测试也传数据集是用于指导模型选择何种输出头
        },
        # 加载指定的部分模块的微调权重。
        "finetune_checkpoint": {
            # "tts_model": "./unitaf_ckpt/UniTAF-A2F(lr_1e-4)- LoRA-TTS(lr_5e-7_rank_128)/checkpoint-20000/tts_model.pt",
            "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F(口型状态顶点空间加权loss_加权仅作用于嘴部顶点)-加载Adapter预训练权重(约束AudioFeature_step_74140)_260109/checkpoint-74140/audio_feature_projector.pt",
            "a2f_model": "./unitaf_ckpt/UniTAF-A2F(口型状态顶点空间加权loss_加权仅作用于嘴部顶点)-加载Adapter预训练权重(约束AudioFeature_step_74140)_260109/checkpoint-74140/a2f_model.pt",
        }
    }
    cfg = OmegaConf.create(train_config)
    unitaf = UniTextAudioFaceModel(cfg, device=torch.device("cuda:0"), mode="inference")  # 推理模式

    # 执行测试
    test_unitaf_stream(unitaf)