import numpy as np
import pickle
import torch
from dataclasses import dataclass
from smplx.lbs import lbs
from smplx.utils import Struct, to_np, to_tensor
from torch import nn


@dataclass
class FlameParams:
    betas: torch.Tensor
    jaw: torch.Tensor
    eyeballs: torch.Tensor
    neck: torch.Tensor


class FlameLayer(nn.Module):

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.dtype = torch.float32
        with open(model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer('shapedirs',
                             to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(
            to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)
        self.register_buffer(
            'v_template',
            to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype),
        )
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs,
                              [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))
        default_neck_pose = torch.zeros([1, 3],
                                        dtype=self.dtype,
                                        requires_grad=False)
        self.register_parameter(
            'neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        default_eyeball_pose = torch.zeros([1, 6],
                                           dtype=self.dtype,
                                           requires_grad=False)
        self.register_parameter(
            'eyeballs',
            nn.Parameter(default_eyeball_pose, requires_grad=False))
        default_jaw = torch.zeros([1, 3],
                                  dtype=self.dtype,
                                  requires_grad=False)
        self.register_parameter('jaw',
                                nn.Parameter(default_jaw, requires_grad=False))

        self.FLAME_CONSTS = {
            'betas': 400,
            'rotation': 6,
            'jaw': 3,
            'eyeballs': 0,
            'neck': 0,
            'translation': 3,
            'scale': 1,
        }

    def flame_params_from_3dmm(self,
                               tensor_3dmm: torch.Tensor,
                               zero_expr: bool = False):
        constants = self.FLAME_CONSTS

        assert tensor_3dmm.ndim == 2

        cur_index: int = 0
        betas = tensor_3dmm[:, :constants['betas']]
        cur_index += constants['betas']
        jaw = tensor_3dmm[:, cur_index:cur_index + constants['jaw']]
        cur_index += constants['jaw']
        cur_index += constants['rotation']
        eyeballs = tensor_3dmm[:, cur_index:cur_index + constants['eyeballs']]
        cur_index += constants['eyeballs']
        neck = tensor_3dmm[:, cur_index:cur_index + constants['neck']]
        cur_index += constants['neck']
        cur_index += constants['translation']
        cur_index += constants['scale']
        return FlameParams(betas=betas, jaw=jaw, eyeballs=eyeballs, neck=neck)

    def forward(self, tensor_3dmm: torch.Tensor):
        bs = tensor_3dmm.shape[0]
        flame_params = self.flame_params_from_3dmm(tensor_3dmm)
        betas = flame_params.betas
        rotation = torch.zeros([bs, 3], device=tensor_3dmm.device)
        neck_pose = flame_params.neck if not (
            0 in flame_params.neck.shape) else self.neck_pose[[0]].expand(
                bs, -1)
        eyeballs = flame_params.eyeballs if not (
            0 in flame_params.eyeballs.shape) else self.eyeballs[[0]].expand(
                bs, -1)
        jaw = flame_params.jaw if not (
            0 in flame_params.jaw.shape) else self.jaw[[0]].expand(bs, -1)

        full_pose = torch.cat([rotation, neck_pose, jaw, eyeballs], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(bs, 1, 1)
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )
        return vertices


class FlameParamLossForDADHead(nn.Module):

    def __init__(self, mouth_indices_path: str, loss_mask_path: str,
                 flame_model_path: str) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path)
        loss_indices = np.load(loss_mask_path)
        self.register_buffer('mouth_idx', torch.from_numpy(mouth_indices))
        self.register_buffer('loss_indices', torch.from_numpy(loss_indices))
        self.mse = nn.MSELoss()
        self.flame_layer = FlameLayer(flame_model_path)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        '''
        参数 → 顶点
            把输入的 FLAME 参数 x（shape 任意，最后一维是 100 维或 103 维等）和真值参数 target 分别送进
            self.flame_layer(...)，得到 N×V×3 的完整顶点坐标。
        只挑关心区域
            用预存的 loss_indices（顶点索引掩码）把 V 维顶点切成子集（例如只保留脸、嘴、眼等区域），
            形状变成 [B, T, V', 3]。
        MSE 误差
            对掩码后的顶点坐标直接 MSELoss(x, target)，这就是 训练用的唯一 loss。
        '''
        ori_shape = x.shape
        x = self.flame_layer(x.reshape(-1, ori_shape[-1]))
        target = self.flame_layer(target.reshape(-1, ori_shape[-1]))
        x = x.reshape((ori_shape[0], ori_shape[1], -1, 3))
        target = target.reshape((ori_shape[0], ori_shape[1], -1, 3))
        return self.mse(x[:, :, self.loss_indices], target[:, :,
                                                           self.loss_indices])

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        x = self.flame_layer(x.reshape(-1, x.shape[-1]))
        return x[:, self.loss_indices]

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        x = self.flame_layer(x.reshape(-1, ori_shape[-1]))
        target = self.flame_layer(target.reshape(-1, ori_shape[-1]))
        metric_L2 = (target[:, self.mouth_idx] - x[:, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class VerticesLoss(nn.Module):
    '''
    类中三个方法全部在“顶点空间”完成——输入 x 和 target 被默认已经是 N×V×3 或 N×(V·3) 的顶点坐标。
    forward:
        直接 MSELoss(x, target)，把两个顶点序列逐元素做均方误差，这就是整条训练损失。
    get_vertices:
        只做 reshape(-1, x.shape[-1]//3, 3)，把最后一维展开成 [N, V, 3] 格式，方便后续可视化或度量，本身不算 loss。
    mouth_metric:
        1.先统一 reshape 成 [B, T, V, 3],
        2.用预存的 mouth_idx 只抽口部顶点，
        3.计算口部每个顶点在 x/y/z 上的 L2 误差，帧内求和、口内取 max、再开根号，
            得到 metric_L2norm = max_{mouth}‖v_target−v_pred‖₂
            这一步只用于评测口型同步精度，不反向传播。

    '''
    def __init__(self, mouth_indices_path: str) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path)
        self.register_buffer('mouth_idx', torch.from_numpy(mouth_indices))
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        return x.reshape(-1, x.shape[-1] // 3, 3)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        target = target.reshape(*ori_shape[:-1], -1, 3)
        x = x.reshape(*ori_shape[:-1], -1, 3)
        metric_L2 = (target[:, :, self.mouth_idx] - x[:, :, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class BlendShapeLoss(nn.Module):
    '''
    系数空间（51 维 blendshape 权重）
        普通模式:
            直接 MSE(x, target)，对 51 维（或 61 维）系数做逐元素均方误差。
        Style-control 模式:
            把61维度拆成4组，非关键系数，眨眼[0,1]，下颔开合[19,20]，嘴型[40,41] 分别算 MSE 后按权重相加。
    顶点空间（三维顶点坐标）
        系数loss：
            loss1 = bs_beta * MSE(x, target) 仍在系数空间加权，但权重由 bs_beta 控制。
        顶点loss：
            先把 51 维系数通过线性混合形状公式，还原成 N×3 的顶点，再算 MSE。
    口部度量（mouth_metric）
        只取 口部顶点索引（self.mouth_idx），计算 L2 距离并求 max-over-time 的误差，用于评估口部同步精度，不参与训练反向传播。
    '''
    def __init__(self,
                 mouth_indices_path: str,
                 blendshape_path: str,
                 bs_beta: float = 0.0,
                 components_num: int = None) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path)
        blendshape = np.load(blendshape_path)
        self.register_buffer('mouth_idx', torch.from_numpy(mouth_indices))
        mean_shape = blendshape['meanshape']
        mean_shape = mean_shape.reshape(-1).astype(np.float32)
        blend_shape = blendshape['blendshape']
        blend_shape = blend_shape.reshape(len(blend_shape),
                                          -1).astype(np.float32)
        self.register_buffer('mean_shape', torch.from_numpy(mean_shape))
        self.register_buffer('blend_shape', torch.from_numpy(blend_shape))
        self.bs_beta = bs_beta
        self.mse = nn.MSELoss()
        self.components_num = components_num

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.bs_beta * self.mse(x, target) + self.mse(
            self.bs2vertices(x), self.bs2vertices(target))

    def bs2vertices(self, x: torch.Tensor):
        return x[:, :self.components_num] @ \
            self.blend_shape[:self.components_num] + \
            self.mean_shape

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        x = self.bs2vertices(x)
        return x.reshape(-1, x.shape[-1] // 3, 3)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        target = self.bs2vertices(target)
        x = self.bs2vertices(x)
        ori_shape = x.shape
        target = target.reshape(*ori_shape[:-1], -1, 3)
        x = x.reshape(*ori_shape[:-1], -1, 3)
        metric_L2 = (target[:, :, self.mouth_idx] - x[:, :, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class ParamLoss(nn.Module):

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
        self.L1 = nn.L1Loss()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.weight * self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target), self.L1(x, target)

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        return x.squeeze(0)


class UniTalkerLoss(nn.Module):
    '''
    每个数据集/标注类型有专用的损失计算。
    使得UniTalker能够统一处理多种不同类型的面部动画数据，从混合形状参数到3D顶点坐标，为多数据集训练和灵活的推理输出提供了基础。
    '''
    def __init__(self, args) -> None:
        super().__init__()
        loss_config = {
            'flame_params_from_dadhead': {
                'class': FlameParamLossForDADHead,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/02_flame_mouth_idx.npy',
                    'loss_mask_path':
                    'a2f/resources/binary_resources/flame_humanface_index.npy',
                    'flame_model_path': 'a2f/resources/binary_resources/flame.pkl',
                }
            },
            '3DETF_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/03_emo_talk_mouth_idx.npy',
                    'blendshape_path':
                    'a2f/resources/binary_resources/EmoTalk.npz',
                    'bs_beta': args.blendshape_weight,
                },
            },
            'inhouse_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/05_inhouse_arkit_mouth_idx.npy',
                    'blendshape_path':
                    'a2f/resources/binary_resources/inhouse_arkit.npz',
                    'bs_beta': args.blendshape_weight,
                },
            },
            'FLAME_5023_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/02_flame_mouth_idx.npy',
                },
            },
            'BIWI_23370_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/04_BIWI_mouth_idx.npy',
                },
            },
            'FLAME_SUB_2055_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/mouth_idx_FLAME_5023_vertices_wo_eyes_wo_head.npy',
                },
            },
            'meshtalk_6172_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    'a2f/resources/binary_resources/06_mesh_talk_mouth_idx.npy',
                },
            },
            '24_viseme': {
                'class': ParamLoss,
                'args': {
                    'weight': args.blendshape_weight
                },
            }
        }

        loss_module_dict = {}
        for k, v in loss_config.items():
            loss_module = v['class'](**v['args'])
            loss_module_dict[k] = loss_module
        self.loss_module_dict = nn.ModuleDict(loss_module_dict)
        self.mse = nn.MSELoss()
        return

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        annot_type: str,
    ):
        return self.loss_module_dict[annot_type](x, target)

    def pca_loss(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor,
                     annot_type: str):
        return self.loss_module_dict[annot_type].mouth_metric(x, target)

    def get_vertices(self, x: torch.Tensor, annot_type: str):
        return self.loss_module_dict[annot_type].get_vertices(x)