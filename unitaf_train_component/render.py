"""Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V.

(MPG) is holder of all proprietary rights on this computer program. You can
only use this computer program if you have closed a license agreement with MPG
or you get the right to use the computer program from someone who is authorized
to grant you that right. Any use of the computer program without a valid
license is prohibited and liable to prosecution. Copyright 2019 Max-Planck-
Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of
its Max Planck Institute for Intelligent Systems and the Max Planck Institute
for Biological Cybernetics. All rights reserved. More information about VOCA is
available at http://voca.is.tue.mpg.de. For comments or questions, please email
us at voca@tue.mpg.de
/
版权声明：Max-Planck-Gesellschaft（马普学会）拥有本计算机程序的全部专有权利。
只有与 MPG 签订许可协议或获得授权后才能使用；未经授权使用将被追究法律责任。
更多信息见 http://voca.is.tue.mpg.de，疑问请联系 voca@tue.mpg.de
版权 2019，Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V.
"""

import cv2
import numpy as np
import os
import subprocess
import torch
from tqdm import tqdm
from unitaf_train.unitaf_dataset_support_config import unitaf_dataset_support_config

def render_meshes(mesh_vertices: torch.Tensor,
                  faces: torch.Tensor,
                  img_size: tuple = (256, 256),
                  aa_factor: int = 1,
                  vis_bs: int = 100):
    '''
    函数：render_meshes
    功能：把一批三维网格渲染成二维图像（RGB 或 RGBA）
    参数：
      mesh_vertices: [N, V, 3] 顶点坐标
      faces:         [N, F, 3] 面片索引，或 [1, F, 3] 复用同一份面
      img_size:      输出图像分辨率 (宽, 高)
      aa_factor:     抗锯齿超采样倍数，>1 时先大图渲染再缩小
      vis_bs:        每次送入 GPU 的网格数量，防止显存爆炸
    返回： [N, H, W, 3/4] 的 uint8 图像张量（0-255）
    '''
    # 合法性检查：要么每套顶点对应一套面，要么所有顶点共用一套面
    assert len(mesh_vertices) == len(faces) or len(faces) == 1
    if len(faces) == 1:   # 如果只有一套面，则复制 N 份
        faces = faces.repeat(len(mesh_vertices), 1, 1)

    # 计算顶点在 x/y 方向的最大绝对值，用于后续正交相机范围
    x_max = y_max = mesh_vertices[..., 0:2].abs().max()
    from pytorch3d.renderer import (
        DirectionalLights,
        FoVOrthographicCameras,
        HardPhongShader,
        Materials,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.renderer.blending import BlendParams
    from pytorch3d.renderer.materials import Materials
    from pytorch3d.structures import Meshes

    aa_factor = int(aa_factor)  # 超采样倍数取整
    device = mesh_vertices.device  # 与输入顶点同一 GPU
    verts_batch_lst = torch.split(mesh_vertices, vis_bs)  # 按 vis_bs 分块
    faces_batch_lst = torch.split(faces, vis_bs)  # 面也对应分块

    # 构造相机外参：旋转矩阵 R 与平移向量 T
    R = torch.eye(3).to(mesh_vertices.device)
    R[0, 0] = -1  # 左右翻转
    R[2, 2] = -1  # 前后翻转
    T = torch.zeros(3).to(mesh_vertices.device)
    T[2] = 10     # 把相机往后移 10 个单位
    R, T = R[None], T[None]  # 增加 batch 维度
    W, H = img_size   # 目标图像宽高
    # 正交相机：视景体由 min/max_x/y 定义，z 范围 0.01~3
    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=0.01,
        zfar=3,
        max_x=x_max * 1.2,
        min_x=-x_max * 1.2,
        max_y=y_max * 1.2,
        min_y=-y_max * 1.2,
    )
    # cameras = FoVPerspectiveCameras(
    # 	device=device,
    # 	R=R,
    # 	T=T,
    # 	znear=0.01,
    # 	zfar=3,
    # 	aspect_ratio=1,
    # 	fov=0.3,
    # 	degrees=False
    # 	)

    # 光栅化设置：图像尺寸=目标尺寸*超采样倍数，每个像素只取最近面
    raster_settings = RasterizationSettings(
        image_size=(W * aa_factor, H * aa_factor),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    raster = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )
    # 混合参数：背景纯黑
    blend_params = BlendParams(
        sigma=0.0, gamma=0.0, background_color=(0.0, 0.0, 0.0))
    # 平行光：方向沿 +Z，环境光 0.3，漫反射 0.6，镜面 0.1
    lights = DirectionalLights(
        device=device,
        direction=((0, 0, 1), ),
        ambient_color=((0.3, 0.3, 0.3), ),
        diffuse_color=((0.6, 0.6, 0.6), ),
        specular_color=((0.1, 0.1, 0.1), ))
    # 材质：全白色，镜面反射强度 15
    materias = Materials(
        ambient_color=((1, 1, 1), ),
        diffuse_color=((1, 1, 1), ),
        specular_color=((1, 1, 1), ),
        shininess=15,
        device=device)
    # 着色器：硬 Phong 模型
    shader = HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materias,
        blend_params=blend_params)
    # 渲染器 = 光栅化 + 着色
    renderer = MeshRenderer(
        rasterizer=raster,
        shader=shader,
    )

    rendered_imgs = []  # 收集各批次图像
    for verts_batch, faces_batch in tqdm(
            zip(verts_batch_lst, faces_batch_lst),
            total=(len(verts_batch_lst))):
        # 顶点颜色纹理：全白
        textures = TexturesVertex(verts_features=torch.ones_like(verts_batch))
        # 构造 Meshes 对象
        meshes = Meshes(
            verts=verts_batch, faces=faces_batch, textures=textures)
        with torch.no_grad():
            imgs = renderer(meshes)  # 真正渲染，得到 N×H×W×4
        rendered_imgs.append(imgs.cpu())  # 搬回内存
    rendered_imgs = torch.cat(rendered_imgs, dim=0)  # 合并所有批次

    # 如果开了抗锯齿，先 NHWC→NCHW，再用双三次下采样，最后 NCHW→NHWC
    if aa_factor > 1:
        rendered_imgs = rendered_imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW
        rendered_imgs = torch.nn.functional.interpolate(
            rendered_imgs, scale_factor=1 / aa_factor, mode='bicubic')
        rendered_imgs = rendered_imgs.permute(0, 2, 3, 1)  # NCHW -> NHWC
    return rendered_imgs


def read_obj(in_path):
    '''
    读取简易 .obj 文件，返回顶点与面（索引从 0 开始）
    '''
    with open(in_path, 'r') as obj_file:
        # Read the lines of the OBJ file
        lines = obj_file.readlines()

    # Initialize empty lists for vertices and faces
    verts = []  # 存放顶点
    faces = []  # 存放面
    for line in lines:
        line = line.strip()  # 去首尾空白 / Remove leading/trailing whitespace
        elements = line.split()  # 按空格分割 / Split the line into elements

        if len(elements) == 0:
            continue  # 跳过空行 / Skip empty lines

        # Check the type of line (vertex or face)
        if elements[0] == 'v':  # 顶点行
            x, y, z = map(float,
                          elements[1:4])  # Extract the vertex coordinates
            verts.append((x, y, z))  # Add the vertex to the list
        elif elements[0] == 'f':  # 面行
            # 支持 "f v/vt/vn" 格式，只取顶点索引
            face_indices = [
                int(index.split('/')[0]) for index in elements[1:]
            ]  # Extract the vertex indices
            faces.append(face_indices)  # Add the face to the list
    return np.array(verts), np.array(faces)


def pad_for_libx264(image_array):
    """
    如果图像高或宽不是偶数，则用 0 填充到偶数，避免 libx264 报错
    支持 2~4 维数组：单张灰度、单张彩色、批量灰度、批量彩色
    /
    Pad zeros if width or height of image_array is not divisible by 2.
    Otherwise you will get.

    \"[libx264 @ 0x1b1d560] width not divisible by 2 \"

    Args:
            image_array (np.ndarray):
                    Image or images load by cv2.imread().
                    Possible shapes:
                    1. [height, width]
                    2. [height, width, channels]
                    3. [images, height, width]
                    4. [images, height, width, channels]

    Returns:
            np.ndarray:
                    A image with both edges divisible by 2.
    """
    # 根据维度判断高、宽所在的 axis
    if image_array.ndim == 2 or \
      (image_array.ndim == 3 and image_array.shape[2] == 3):
        hei_index = 0
        wid_index = 1
    elif image_array.ndim == 4 or \
      (image_array.ndim == 3 and image_array.shape[2] != 3):
        hei_index = 1
        wid_index = 2
    else:
        return image_array   # 其他形状直接返回

    hei_pad = image_array.shape[hei_index] % 2  # 需要补的行数
    wid_pad = image_array.shape[wid_index] % 2  # 需要补的列数
    if hei_pad + wid_pad > 0: # 只有出现奇数才补
        pad_width = []  # 构造 np.pad 用的参数
        for dim_index in range(image_array.ndim):
            if dim_index == hei_index:
                pad_width.append((0, hei_pad))
            elif dim_index == wid_index:
                pad_width.append((0, wid_pad))
            else:
                pad_width.append((0, 0))
        values = 0
        image_array = \
         np.pad(image_array,
             pad_width,
             mode='constant', constant_values=values)  # 常数 0 填充
    return image_array


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps=30,
    resolution=None,   # 可强制指定输出分辨率
    disable_log: bool = False,
) -> None:
    """
    把 [帧数, 高, 宽, 3] 的 uint8 数组直接写成 H.264 mp4
    依赖系统 ffmpeg，默认使用 libx264 编码，无音频
    /
    Convert an array to a video directly, gif not supported.

    Args:
            image_array (np.ndarray): shape should be (f * h * w * 3).
            output_path (str): output video file path.
            fps (Union[int, float, optional): fps. Defaults to 30.
            resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                    optional): (height, width) of the output video.
                    Defaults to None.
            disable_log (bool, optional): whether close the ffmepg command info.
                    Defaults to False.
    Raises:
            FileNotFoundError: check output path.
            TypeError: check input array.

    Returns:
            None.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4  # 必须是 NHWC
    assert image_array.shape[-1] == 3  # 必须是 3 通道
    # 如果手动指定分辨率
    if resolution:
        height, width = resolution
        width += width % 2  # 保证偶数
        height += height % 2
    else:  # 否则自动 pad 成偶数
        image_array = pad_for_libx264(image_array)
        height, width = image_array.shape[1], image_array.shape[2]

    # 构造 ffmpeg 命令行：rawvideo → libx264
    command = [
        '/usr/bin/ffmpeg',  # 可执行路径
        '-y',  # 覆盖输出 / (optional) overwrite output file if it exists
        '-f', 'rawvideo',  # 输入格式
        '-s', f'{int(width)}x{int(height)}',  # 单帧尺寸 / size of one frame
        '-pix_fmt', 'bgr24',        # OpenCV 默认 BGR
        '-r', f'{fps}',             # 帧率 / frames per second
        '-loglevel', 'error',       # 只打印错误
        '-threads', '4',            # 并行编码
        '-i', '-',                  # 从 stdin 读帧 / The input comes from a pipe
        '-vcodec', 'libx264',       # 视频编码器
        '-an',                      # 无音频 / Tells FFMPEG not to expect any audio
        output_path,
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    # 启动子进程，建立管道
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError('No buffer received.')
    # 逐帧写入 stdin
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()   # 等待编码完成


def get_obj_faces(annot_type: str):
    '''
    根据数据集类型，读取对应模板 obj 的面片索引（返回 0-base）
    '''
    template_obj_path_dict = {
        '3DETF_blendshape_weight': 'resources/obj_template/3DETF_blendshape_weight.obj',
        'FLAME_5023_vertices': 'resources/obj_template/FLAME_5023_vertices.obj',
        'BIWI_23370_vertices': 'resources/obj_template/BIWI_23370_vertices.obj',
        'flame_params_from_dadhead': 'resources/obj_template/flame_params_from_dadhead.obj',
        'inhouse_blendshape_weight': 'resources/obj_template/inhouse_blendshape_weight.obj',
        'meshtalk_6172_vertices': 'resources/obj_template/meshtalk_6172_vertices.obj'
    }
    # 读 obj 取面，如果 obj 是 1-base 则转 0-base
    faces = np.array(read_obj(template_obj_path_dict[annot_type])[1])
    if faces.min() == 1:
        faces = faces - 1
    return faces


def vis_model_out_proxy():
    '''
    命令行入口，把模型输出的 npz 文件批量渲染成视频（带音频）
    用法： python this_script.py  <out.npz>  <audio_dir>  <out_dir>
    '''
    import sys
    npz_path = sys.argv[1]      # 模型输出 npz
    audio_dir = sys.argv[2]     # 原始 wav 所在目录
    out_dir = sys.argv[3]       # 视频保存目录
    save_images = False         # True 则保存逐帧 png，False 保存 mp4

    os.makedirs(out_dir, exist_ok=True)
    out_dict = np.load(npz_path, allow_pickle=True)
    print(list(out_dict.keys()))     # 打印 npz 包含的 wav 文件名列表

    # 遍历每个 wav 的预测结果
    for wav_f in out_dict.keys():
        wav_path = os.path.join(audio_dir, wav_f)
        cur_out_dict = out_dict[wav_f]      # 取出该 wav 的所有数据集结果
        for datasetname, out in cur_out_dict.item().items():
            out = torch.Tensor(out)         # numpy → torch
            annot_type = unitaf_dataset_support_config[datasetname]['annot_type']  # 数据集类型
            faces = get_obj_faces(annot_type)   # 读取对应模板面片

            with torch.no_grad():
                if save_images:
                    out_size = (1024, 1024)     # 保存图片用高分辨率
                else:
                    out_size = (512, 512)       # 视频用 512
                # 渲染网格 → 图像数组
                img_array = render_meshes(out.cuda(),
                                          torch.from_numpy(faces[None]).cuda(),
                                          out_size)
                if save_images:
                    channels = 4                # 保存 png 带 alpha
                else:
                    channels = 3                # 视频只要 RGB
                # 0-1 → 0-255
                img_array = (img_array[..., :channels].cpu().numpy() *
                             255).astype(np.uint8)

            out_key = f'{wav_f[:-4]}'           # 去掉 .wav 后缀当 key

            if save_images:
                # 保存逐帧 png
                out_images_dir = os.path.join(out_dir, datasetname, out_key)
                os.makedirs(out_images_dir, exist_ok=True)
                for imgidx, img in enumerate(img_array):
                    img_path = os.path.join(out_images_dir,
                                            f'{imgidx:04d}.png')
                    cv2.imwrite(img_path, img)
                continue

            # 否则保存视频
            out_video_path = os.path.join(out_dir,
                                          f'{out_key}_{datasetname}.mp4')
            out_video_dir = os.path.dirname(out_video_path)
            os.makedirs(out_video_dir, exist_ok=True)
            tmp_path = os.path.join(out_video_dir, 'tmp.mp4')  # 先写无声视频
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            # 不同数据集帧率不同
            if annot_type == 'BIWI_23370_vertices':
                fps = 25
            else:
                fps = 30
            print(img_array.shape, img_array.dtype)
            array_to_video(img_array, output_path=tmp_path, fps=fps)

            # 二次调用 ffmpeg 把音频混流进去，-shortest 取最短长度
            cmd = f'ffmpeg -i {tmp_path} -i  {wav_path} -c:v copy -c:a aac -shortest -strict -2  {out_video_path} -y '
            print(cmd)
            subprocess.run(cmd, shell=True)
            os.remove(tmp_path)   # 删除临时无声视频

# 当脚本被直接运行时，执行 vis_model_out_proxy
if __name__ == '__main__':
    vis_model_out_proxy()
