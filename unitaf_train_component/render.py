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
import io
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import subprocess
import torch
import torchaudio
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

# FIXME：这里实现失败，存在bug，后面还是安装pytorch3d以直接使用render_meshes而非render_meshes_np
def render_meshes_np(mesh_vertices: torch.Tensor,
                     faces: torch.Tensor,
                     img_size: tuple = (256, 256),
                     aa_factor: int = 1,
                     vis_bs: int = 100):
    """
    纯 numpy + torch 渲染，无需 pytorch3d。
    参数与原来 render_meshes 完全一致，返回 [N, H, W, 3] uint8 0-255
    仅支持：正交投影 + Lambert 漫反射 + 纯色背景
    """
    if faces.dim() == 3:  # [N, F, 3] 或 [1, F, 3]
        faces = faces[0]  # 取第 0 份，复用同一份面

    device = mesh_vertices.device
    N, V, _ = mesh_vertices.shape
    F, _ = faces.shape
    H, W = img_size
    # 如果只有一套面，复制 N 份
    if faces.dim() == 2 and faces.shape[0] == F:
        faces = faces.expand(N, -1, 3)

    # 1. 相机参数（正交）
    x_max = mesh_vertices[..., 0].abs().max().item()
    y_max = mesh_vertices[..., 1].abs().max().item()
    scale = max(x_max, y_max) * 1.2
    # 顶点 -> 屏幕坐标 [-1,1]
    verts_2d = mesh_vertices[..., :2] / scale  # [N,V,2]

    # 2. 光照方向（固定）
    light_dir = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)

    # 3. 逐批绘制
    imgs = []
    for i in range(0, N, vis_bs):
        end = min(i + vis_bs, N)
        v2d = verts_2d[i:end]  # [bs, V, 2]
        v3d = mesh_vertices[i:end]  # [bs, V, 3]
        f = faces[i:end]  # [bs, F, 3]
        bs = v2d.shape[0]

        # 3-1 光栅化：自己写“z-buffer”
        # 先初始化深度图、颜色图
        depth = torch.full((bs, H, W), 1e6, device=device)
        normal_map = torch.zeros((bs, H, W, 3), device=device)
        mask = torch.zeros((bs, H, W), dtype=torch.bool, device=device)

        for b in range(bs):
            # 把三角面投影到像素坐标
            face_2d = (v2d[b][f[b]] + 1) * 0.5  # [F,3,2]  0-1
            face_2d = face_2d * torch.tensor([W, H], device=device)  # -> 像素
            face_2d = face_2d.long()

            # 简单扫描线：取 bbox 再判断重心坐标
            for f_idx in range(f.shape[1]):
                tri = face_2d[f_idx]  # [3, 2]
                x_min, x_max = tri[:, 0].min(), tri[:, 0].max()
                y_min, y_max = tri[:, 1].min(), tri[:, 1].max()
                if x_max < 0 or x_min >= W or y_max < 0 or y_min >= H:
                    continue
                x_min = max(x_min.item(), 0)
                x_max = min(x_max.item() + 1, W)
                y_min = max(y_min.item(), 0)
                y_max = min(y_max.item() + 1, H)

                vs = v3d[b][f[b][f_idx]].float()  # [3,3]
                v0, v1, v2 = vs[0], vs[1], vs[2]
                normal = torch.cross(v1 - v0, v2 - v0)
                normal = F.normalize(normal, dim=0)

                yy, xx = torch.meshgrid(
                    torch.arange(y_min, y_max, device=device),
                    torch.arange(x_min, x_max, device=device),
                    indexing='ij')
                pix = torch.stack([xx, yy], dim=-1)  # [H', W', 2]

                # 重心坐标判内点
                def _wedge(a, b):
                    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

                v0v1 = tri[1] - tri[0]
                v0v2 = tri[2] - tri[0]
                area = _wedge(v0v1, v0v2) + 1e-6
                pv = pix - tri[0]
                u = _wedge(pv, v0v2) / area
                v = _wedge(v0v1, pv) / area
                w = 1 - u - v
                inside = (u >= 0) & (v >= 0) & (w >= 0)
                if inside.sum() == 0:
                    continue

                # 计算像素深度（z-buffer）
                z_pix = w * vs[0, 2] + u * vs[1, 2] + v * vs[2, 2]
                update = inside & (z_pix < depth[b, y_min:y_max, x_min:x_max])
                depth[b, y_min:y_max, x_min:x_max][update] = z_pix[update]
                mask[b, y_min:y_max, x_min:x_max][update] = True
                normal_map[b, y_min:y_max, x_min:x_max][update] = normal

        # 3-2 Lambert 光照
        lambert = (normal_map * light_dir).sum(dim=-1).clamp(min=0)  # [bs, H, W]
        color = 0.9 * lambert + 0.1  # 环境光 0.1
        color = color.unsqueeze(-1).expand(-1, -1, -1, 3)  # -> [bs, H, W, 3]
        bg = torch.zeros_like(color)
        rgb = torch.where(mask.unsqueeze(-1), color, bg)  # 背景纯黑

        # 3-3 抗锯齿（简化版：先 2× 再插值回目标尺寸）
        if aa_factor > 1:
            rgb = rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
            rgb = F.interpolate(rgb, scale_factor=1 / aa_factor, mode='bilinear', align_corners=False)
            rgb = rgb.permute(0, 2, 3, 1)  # BCHW -> BHWC

        imgs.append((rgb * 255).cpu().numpy().astype(np.uint8))

    return np.concatenate(imgs, axis=0)  # [N, H, W, 3]

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
    codec: str = "libx264",          # 指定编码器
    pixel_format: str = "yuv420p"    # 兼容性最好的像素格式
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
        '-vcodec', codec,           # 视频编码器
        '-pix_fmt', pixel_format,   # 像素格式
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
        '3DETF_blendshape_weight': 'a2f/resources/obj_template/3DETF_blendshape_weight.obj',
        'FLAME_5023_vertices': 'a2f/resources/obj_template/FLAME_5023_vertices.obj',
        'BIWI_23370_vertices': 'a2f/resources/obj_template/BIWI_23370_vertices.obj',
        'flame_params_from_dadhead': 'a2f/resources/obj_template/flame_params_from_dadhead.obj',
        'inhouse_blendshape_weight': 'a2f/resources/obj_template/inhouse_blendshape_weight.obj',
        'meshtalk_6172_vertices': 'a2f/resources/obj_template/meshtalk_6172_vertices.obj',
        'qxsk_inhouse_blendshape_weight': 'a2f/resources/obj_template/inhouse_blendshape_weight.obj',
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

def render_npz_video(out_np: np.ndarray,          # [T, V, 3] 顶点序列
                     audio_path: str,              # 完整 wav 路径；None/"" 表示无声
                     out_dir: str,                 # 输出根目录
                     annot_type: str,              # 显式传入模板类型，如 "qxsk_inhouse_blendshape_weight"
                     save_images: bool = False,    # False=mp4，True=逐帧png
                     device: str = "cuda"):
    """
    把「单个音频」对应的顶点序列直接渲染成视频（或逐帧png）。
    与之前最大区别：不再遍历 npz 字典，也不再自动查表 annot_type，由调用者一次性给齐。
    """
    from pathlib import Path

    # ---- 1. 准备目录 ----
    os.makedirs(out_dir, exist_ok=True)
    # # 读音频长度
    # if audio_path and Path(audio_path).exists():
    #     info = torchaudio.info(audio_path)
    #     print(f"Audio音频采样率：{info.sample_rate}")  # 22050
    #     print(f"Audio音频总帧数：{info.num_frames}")  # 232448
    #     print(f"Audio音频长度：{info.num_frames / info.sample_rate} 秒")  # 10.541859410430838 秒

    # ---- 2. 顶点转 Torch ----
    verts = torch.Tensor(out_np).to(device)          # [T, V, 3]
    T = verts.shape[0]
    # print(f"Face表情总帧数：{T}")  # 265
    # print(f"Face表情在25fps下的时长：{T / 25}") # 10.6

    # ---- 3. 读面片 ----
    faces = get_obj_faces(annot_type)                # [F, 3] numpy
    faces = torch.from_numpy(faces[None]).to(device) # [1, F, 3]

    # ---- 4. 渲染 ----
    img_size = (1024, 1024) if save_images else (512, 512)
    with torch.no_grad():
        # 一般这里使用pytorch3d的render_meshes实现，如果pytorch3d无法安装，则尝试使用render_meshes_np
        img_array = render_meshes(verts, faces, img_size)  # [T, H, W, 4] float 0-1
        channels = 4 if save_images else 3
        img_array = (img_array[..., :channels].cpu().numpy() * 255).astype(np.uint8)

    # ---- 5. 文件主名（取音频文件名） ----
    out_key = Path(audio_path).stem if audio_path else "demo"  # 去掉路径和后缀

    # ---- 6. 分支：png 序列 ----
    if save_images:
        png_dir = Path(out_dir) / annot_type / out_key
        png_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(img_array):
            cv2.imwrite(str(png_dir / f"{idx:05d}.png"), img)
        print(f">> 逐帧 png 已保存到 {png_dir}")
        return

    # ---- 7. 分支：mp4 视频 ----
    '''
    这里fps指定25，因为UniTAF训练中A2F部分以固定25fps训练
    '''
    tmp_path = str(Path(out_dir) / "tmp.mp4")
    # fps = 25 if annot_type == "qxsk_inhouse_blendshape_weight" else 30
    fps = 25
    array_to_video(img_array, tmp_path, fps=fps)

    final_path = str(Path(out_dir) / f"{out_key}_{annot_type}.mp4")
    if os.path.isfile(final_path):
        os.remove(final_path)

    # ---- 8. 混音 or 无声 ----
    if audio_path and Path(audio_path).exists():
        cmd = (f'ffmpeg -i {tmp_path} -i {audio_path} -c:v copy -c:a aac '
               f'-shortest -strict -2 {final_path} -y')
    else:
        cmd = f'ffmpeg -i {tmp_path} -c:v copy {final_path} -y'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp_path)
    print(f">> 视频已生成：{final_path} ({T} 帧, 约 {T/fps:.2f} s)")

def render_multi_window_video(
    verts_dict: dict,       # dict: {'mouth': np.ndarray, 'expr': np.ndarray, 'final': np.ndarray}
    audio_path: str,
    out_dir: str,
    annot_type: str,
    device: str = "cuda",
    window_names: list = ["mouth", "expression", "final"],
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_height: int = 20,   # 上方文字区域高度
):
    """
    将多个顶点序列渲染成多窗口对比视频
    用于比对 口型，表情，最终合并的渲染适配
    verts_dict: dict, key -> 顶点序列 np.ndarray [T, V, 3]
    """
    from pathlib import Path
    import cv2
    import torch
    import numpy as np
    import os, subprocess

    os.makedirs(out_dir, exist_ok=True)

    # 1. 渲染每个顶点序列
    img_arrays = {}
    for key in window_names:
        verts_np = verts_dict[key]
        verts = torch.Tensor(verts_np).to(device)
        faces = torch.from_numpy(get_obj_faces(annot_type)[None]).to(device)
        imgs = render_meshes(verts, faces, (512, 512))  # [T, H, W, 4]
        imgs = (imgs[..., :3].cpu().numpy() * 255).astype(np.uint8)  # RGB

        # 在每帧上方增加文字区域
        T, H, W, C = imgs.shape
        canvas_imgs = []
        for t in range(T):
            canvas = np.zeros((H + text_height, W, C), dtype=np.uint8)
            canvas[text_height:, :, :] = imgs[t]
            cv2.putText(canvas, key, (10, int(text_height * 0.8)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        font_thickness, cv2.LINE_AA)
            canvas_imgs.append(canvas)
        img_arrays[key] = np.array(canvas_imgs)

    # 2. 检查帧数一致
    T = min([v.shape[0] for v in img_arrays.values()])
    for key in img_arrays:
        img_arrays[key] = img_arrays[key][:T]

    # 3. 水平拼接
    concat_imgs = []
    for t in range(T):
        imgs_row = [img_arrays[key][t] for key in window_names]
        concat_imgs.append(np.concatenate(imgs_row, axis=1))
    concat_imgs = np.array(concat_imgs)  # [T, H, W_total, 3]

    # 4. 输出文件
    out_key = Path(audio_path).stem if audio_path else "demo"
    tmp_path = str(Path(out_dir) / f"{out_key}_tmp.mp4")
    final_path = str(Path(out_dir) / f"{out_key}_multiwindow.mp4")
    if os.path.isfile(final_path):
        os.remove(final_path)

    # 5. 写入临时视频
    fps = 25
    array_to_video(concat_imgs, tmp_path, fps=fps)

    # 6. 混音 / 无声
    if audio_path and os.path.exists(audio_path):
        cmd = (f'ffmpeg -i {tmp_path} -i {audio_path} -c:v copy -c:a aac '
               f'-shortest -strict -2 {final_path} -y')
    else:
        cmd = f'ffmpeg -i {tmp_path} -c:v copy {final_path} -y'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp_path)
    print(f">> 多窗口视频已生成：{final_path} ({T} 帧, 约 {T/fps:.2f} s)")


def render_video_for_evaluate(
    gt_waveform,
    gt_motion,              # [T, V, 3]
    out_motion,             # [T, V, 3]
    output_video_path,      # 输出 mp4 完整路径
    annot_type,
    device: str = "cuda",
):
    '''
    并排可视化 GT vs Pred 顶点动画，带音频。
    帧数不一致时自动对齐（补帧或截断）。
    （这里表情传进来的时候就是为顶点数据的tensor，音频为波形）
    全程 tensor 操作，无中间 numpy 落地；音频走内存 pipe。
    '''
    T1, V, _ = gt_motion.shape
    T2, _, _ = out_motion.shape

    # ---- 0. 帧数对齐 ----
    if T2 < T1:  # 预测短了，末尾补最后一帧
        pad = out_motion[-1:].repeat(T1 - T2, 1, 1)
        out_motion = torch.cat([out_motion, pad], dim=0)
    elif T2 > T1:  # 预测长了，直接截断
        out_motion = out_motion[:T1]
    T = T1


    # ---- 1. 面片 ----
    faces = get_obj_faces(annot_type)  # [F, 3] numpy
    faces = torch.from_numpy(faces[None]).to(device)  # [1, F, 3]

    # ---- 2. 渲染 ----
    H, W = 512, 512
    bar_h = 80  # 顶部文字条高度
    with torch.no_grad():
        gt_img = render_meshes(gt_motion, faces, (H, W))[..., :3]       # [T, H, W, 3] float 0-1
        pred_img = render_meshes(out_motion, faces, (H, W))[..., :3]    # [T, H, W, 3]

    # ---- 3. 组装画面 ----
    # 左右拼接
    side = torch.cat([gt_img, pred_img], dim=2)  # [T,H,2W,3]
    # 顶部文字条
    bar = torch.ones((bar_h, 2 * W, 3), dtype=torch.uint8, device='cpu') * 255
    bar = bar.numpy()
    cv2.putText(bar, "Ground-Truth", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(bar, "Prediction", (W + 40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    bar = torch.from_numpy(bar).to(gt_img.device)  # 回到 GPU
    bar = bar.unsqueeze(0).expand(T, -1, -1, -1)  # [T,bar_h,2W,3]
    frames = torch.cat([bar, side], dim=1)  # [T,H+bar_h,2W,3]
    frames = (frames * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

    # ---- 4. 音频：tensor → 临时 wav 文件 ----
    waveform = gt_waveform.cpu()  # [1, N]
    out_dir = Path(output_video_path).parent  # 与 mp4 同目录
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_wav_path = out_dir / f"{Path(output_video_path).stem}_tmp.wav"
    torchaudio.save(str(tmp_wav_path), waveform, sample_rate=24000)

    # ---- 5. 先写无声视频 ----
    tmp_mp4_path = str(out_dir / "tmp.mp4")
    fps = 25  # 想保持与旧代码一致可再判 annot_type
    array_to_video(frames, tmp_mp4_path, fps=fps)  # 你的旧函数

    # ---- 6. 混音 ----
    if Path(tmp_wav_path).exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_mp4_path,
            "-i", tmp_wav_path,
            "-c:v", "copy",  # 视频不再重编
            "-c:a", "aac",
            "-shortest", output_video_path
        ]
    else:
        # 意外无音频，直接拷贝
        cmd = ["ffmpeg", "-y", "-i", tmp_mp4_path, "-c:v", "copy", output_video_path]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ---- 7. 清理临时文件 ----
    os.remove(tmp_mp4_path)
    os.remove(tmp_wav_path)
    print(f">> 对比视频已生成：{output_video_path}")




# 当脚本被直接运行时，执行 vis_model_out_proxy
if __name__ == '__main__':
    '''
    根目录下运行 python -m unitaf_train_component.render
    '''
    # vis_model_out_proxy()  # 当脚本被直接运行时，执行 vis_model_out_proxy

    # 单独渲染的测试
    loaded = np.load("outputs/UniTAF_output.npz", allow_pickle=True)   # 字典
    out_np = loaded['arr_0']                         # 键名默认 arr_0
    # print(f"Face表情的帧数：{len(out_np)}")
    render_npz_video(
        out_np=out_np,
        audio_path="outputs/UniTAF_output.wav",  # 原始 wav 完整路径；None=无声
        out_dir="outputs",  # 想把视频/图片保存文件夹
        annot_type="qxsk_inhouse_blendshape_weight",  # 你当时传给 get_vertices 的同一字符串
        save_images=False,  # False=直接出 mp4；True=出逐帧 png
        device="cuda"  # 或 "cpu"
    )
