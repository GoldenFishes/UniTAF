'''
这里实现一个webui界面，用于手动赋予/操作ARkit表情的各参数值，并且实时渲染以即使查看到表情变化
需要安装pytorch3d
'''
import os
import sys
import gradio as gr
import numpy as np
import torch
from unitaf_train_component.render import get_obj_faces, render_meshes

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
sys.path.insert(0, project_root)

# UniTalker 51维度 ARKit
unitalker_arkit_names =  [
    'EyeBlinkLeft',         # 0
    'EyeBlinkRight',        # 1
    'EyeSquintLeft',        # 2
    'EyeSquintRight',       # 3
    'EyeLookDownLeft',      # 4
    'EyeLookDownRight',     # 5
    'EyeLookInLeft',        # 6
    'EyeLookInRight',       # 7
    'EyeWideLeft',          # 8
    'EyeWideRight',         # 9
    'EyeLookOutLeft',       # 10
    'EyeLookOutRight',      # 11
    'EyeLookUpLeft',        # 12
    'EyeLookUpRight',       # 13

    'BrowDownLeft',         # 14
    'BrowDownRight',        # 15
    'BrowInnerUp',          # 16
    'BrowOuterUpLeft',      # 17
    'BrowOuterUpRight',     # 18

    'JawOpen',              # 19
    'MouthClose',           # 20
    'JawLeft',              # 21
    'JawRight',             # 22
    'JawForward',           # 23

    'MouthUpperUpLeft',     # 24
    'MouthUpperUpRight',    # 25
    'MouthLowerDownLeft',   # 26
    'MouthLowerDownRight',  # 27
    'MouthRollUpper',       # 28
    'MouthRollLower',       # 29
    'MouthSmileLeft',       # 30
    'MouthSmileRight',      # 31
    'MouthDimpleLeft',      # 32
    'MouthDimpleRight',     # 33
    'MouthStretchLeft',     # 34
    'MouthStretchRight',    # 35
    'MouthFrownLeft',       # 36
    'MouthFrownRight',      # 37
    'MouthPressLeft',       # 38
    'MouthPressRight',      # 39
    'MouthPucker',          # 40
    'MouthFunnel',          # 41
    'MouthLeft',            # 42
    'MouthRight',           # 43
    'MouthShrugLower',      # 44
    'MouthShrugUpper',      # 45

    'NoseSneerLeft',        # 46
    'NoseSneerRight',       # 47

    'CheekPuff',            # 48
    'CheekSquintLeft',      # 49
    'CheekSquintRight'      # 50
]

# 苹果官方的
apple_arkit_names = [
            'eyeBlinkLeft', 'eyeLookDownLeft','eyeLookInLeft','eyeLookOutLeft',
            'eyeLookUpLeft','eyeSquintLeft','eyeWideLeft','eyeBlinkRight','eyeLookDownRight','eyeLookInRight',
            'eyeLookOutRight','eyeLookUpRight','eyeSquintRight','eyeWideRight','jawForward',
            'jawLeft','jawRight','jawOpen','mouthClose','mouthFunnel','mouthPucker','mouthRight',
            'mouthLeft','mouthSmileLeft','mouthSmileRight','mouthFrownRight','mouthFrownLeft',
            'mouthDimpleLeft','mouthDimpleRight','mouthStretchLeft','mouthStretchRight','mouthRollLower',
            'mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthPressLeft','mouthPressRight',
            'mouthLowerDownLeft','mouthLowerDownRight','mouthUpperUpLeft','mouthUpperUpRight',
            'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight','cheekPuff',
            'cheekSquintLeft','cheekSquintRight','noseSneerLeft','noseSneerRight','tongueOut'
             ]



class ARkitBlendShapeWebUI():
    def __init__(self):
        '''
        WebUI界面左侧是一个可以滚动的容器，其中能够指定ARkit表情51个参数（可以拖动条）
        当容器汇总参数出现变动时，界面中的51个ARkit参数会整合成 torch.Tensor，
        由bs2vertices和render_single_expr得到最终的img，并渲染在WebUI界面右侧
        '''

        # 参考 a2f/
        mouth_indices = np.load("a2f/resources/binary_resources/05_inhouse_arkit_mouth_idx.npy")
        blendshape = np.load("a2f/resources/binary_resources/inhouse_arkit.npz")
        # print("blendshape:", blendshape)
        mean_shape = blendshape['meanshape'].reshape(-1).astype(np.float32)  # (V*3,)
        blend_shape = blendshape['blendshape']  # 每个ARKit 名称的“顺序”不是 Apple 官方强制的真正的顺序取决于 inhouse_arkit.npz 里 blendshape 的排列
        blend_shape = blend_shape.reshape(len(blend_shape), -1).astype(np.float32)  # (61, V*3)

        self.mean_shape = torch.from_numpy(mean_shape)
        self.blend_shape = torch.from_numpy(blend_shape)
        self.mouth_idx = torch.from_numpy(mouth_indices)

    # bs转顶点以便渲染
    def bs2vertices(self, x: torch.Tensor):
        # 这里components_num为51，只取前51维，因为self.blend_shape权重只有前51维
        # 取最后一维的前 components_num 维
        coeff = x[..., :51]  # (51)
        vertices = coeff @ self.blend_shape[:51] + self.mean_shape
        return vertices  # (V, 3)

    def render_single_expr(self,
                           vertices: torch.Tensor,
                           annot_type: str = "qxsk_inhouse_blendshape_weight",
                           img_size: int = 512,
                           device: str = "cuda"):
        """
        vertices: [V,3] 或 [1,V,3] 的 tensor
        annot_type: 用来拿拓扑
        out_png:  输出 png 完整路径
        """
        # 1. 统一成 [1,V,3]
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)  # [1,V,3]

        # 2. 面片
        faces = get_obj_faces(annot_type)  # numpy [F,3]
        faces = torch.from_numpy(faces[None]).to(device)  # [1,F,3]

        # 3. 渲染
        with torch.no_grad():
            img = render_meshes(vertices, faces, (img_size, img_size))[..., :3]  # [1,H,W,3] 0-1
        img = (img.squeeze(0) * 255).clamp(0, 255).cpu().byte()  # [H,W,3] uint8

        return img

    def compute_mouth_openness_single(self, vertices, min_val = 0.0320, max_val = 0.0450):
        """
        单帧顶点计算嘴巴开合度
        1. 使用 self.mouth_idx 选出口部顶点
        2. 在竖直方向（y 轴）取最大值和最小值
        3. 最大值 - 最小值 = 嘴巴开合度

        参数：
            vertices : Tensor, shape (1, V, 3) 或 (V, 3) 当前帧的 3D 顶点坐标
        返回：
            openness : Float (scalar) 嘴巴开合距离（上下嘴唇距离）
            norm_openness : 嘴巴开合的归一化距离[0,1]区间，0为闭口，1为大张口
        """
        # 如果输入 shape 是 (V,3)，加一个 batch 维度
        if vertices.ndim == 2:
            vertices = vertices.unsqueeze(0)  # shape -> [1,V,3]

        # 选出口部顶点
        mouth_vertices = vertices[:, self.mouth_idx, :]  # shape [1, M, 3]

        # 取竖直方向坐标 (y 轴)
        y_coords = mouth_vertices[..., 1]  # shape [1, M]

        # 计算上下嘴唇差值
        openness = y_coords.max(dim=-1)[0] - y_coords.min(dim=-1)[0]  # shape [1]
        openness = float(openness.squeeze())  # 标量 shape -> scalar

        '''
        对openness进行特殊的归一化，我们认为openness正常的区间为[0.0320,0.0450],将其归一化为[0,1]
        openness小于等于0.0320归一化到0，认为是闭口状态
        openness大于等于0.0450归一化到1，认为是大开口状态
        '''
        # 特殊归一化
        if openness <= min_val:
            norm_openness = 0.0
        elif openness >= max_val:
            norm_openness = 1.0
        else:
            norm_openness = (openness - min_val) / (max_val - min_val)

        return openness, norm_openness

def render_from_sliders(*slider_values):
    """
    slider_values: 51 个 float
    """
    x = torch.tensor(slider_values, dtype=torch.float32, device=device)

    vertices = engine.bs2vertices(x)  # (V*3,)
    vertices = vertices.view(-1, 3)  # (V,3)
    vertices = vertices.to(device)

    # 计算顶点空间中开口大小
    mouth_openness, norm_openness = engine.compute_mouth_openness_single(vertices)

    img = engine.render_single_expr(
        vertices,
        img_size=384,
        device=device
    )

    return img.numpy(), mouth_openness, norm_openness  # gradio 接受 numpy


if __name__ == '__main__':
    '''
    项目根目录下运行：
    python -m unitaf_train_component.render_webui
    '''
    # WebUI实现
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine = ARkitBlendShapeWebUI()
    engine.mean_shape = engine.mean_shape.to(device)
    engine.blend_shape = engine.blend_shape.to(device)

    with gr.Blocks(title="ARKit BlendShape WebUI") as demo:
        gr.HTML("""
            <style>
            #left-panel {
                height: 90vh;
                overflow-y: auto;
                padding-right: 8px;
            }
            #right-panel {
                height: 90vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            </style>
            """)

        with gr.Row():
            # 左侧：可滚动 slider
            with gr.Column(scale=1, elem_id="left-panel"):
                sliders = []
                for i in range(51):
                    sliders.append(
                        gr.Slider(
                            0.0, 1.0,
                            step=0.01,
                            value=0.0,
                            label=unitalker_arkit_names[i],
                        )
                    )

            # 右侧：始终可见渲染
            with gr.Column(scale=1, elem_id="right-panel"):
                img_out = gr.Image(
                    label="Rendered Face",
                    type="numpy",
                    height=512
                )
                mouth_out = gr.Number(
                    label="Mouth Openness",
                    value=0.0,
                    interactive=False  # 禁止交互，仅用于展示结果
                )
                norm_mouth_out = gr.Slider(
                    label="Normalized Mouth Openness, 0为闭口, 1为大张口",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    interactive=False  # 禁止拖动，只用作显示进度
                )


        # 任意 slider 变化 → 触发重新渲染
        for s in sliders:
            s.change(
                fn=render_from_sliders,
                inputs=sliders,
                outputs=[img_out, mouth_out, norm_mouth_out],
            )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)