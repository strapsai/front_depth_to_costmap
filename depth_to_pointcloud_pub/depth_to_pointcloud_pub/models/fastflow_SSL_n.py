# import FrEIA.framework as Ff
# import FrEIA.modules as Fm

import depth_to_pointcloud_pub.fastflow_inn as Ff   # FrEIA 대신
import depth_to_pointcloud_pub.fastflow_inn as Fm

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pspnet

import random

def fix_seed(seed=1):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(1) 

class Normalize(nn.Module) :
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self, input):
        return F.normalize(input, p=2, dim=1)

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=False),
            Normalize()
        )

    return subnet_conv

def nf_fast_flow(input_chw, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

class PSPNET_RADIO_SSL(nn.Module):
    def __init__(
        self,
        in_channels=256,
        flow_steps=8,
        freeze_backbone=True,
        flow='fast',
    ):
        super(PSPNET_RADIO_SSL, self).__init__()

        self.global_extractor = pspnet.radio_encoder_ssl(
            pretrained=True,
            zoom_factor=1,
            final_dim=768,
        )

        self.local_extractor = pspnet.pspnet_encoder_ssl(
            pretrained=False,
            zoom_factor=1,
            final_dim=768,
        )

        self.freeze_backbone = freeze_backbone
        self.input_size = (512, 512)

        self.nf_flows = nf_fast_flow(
            [in_channels, self.input_size[0], self.input_size[1]],
            hidden_ratio=1.0,
            flow_steps=flow_steps,
        )

    def inference(self, input_tensor, proxy):
        self.eval()
        self.global_extractor.eval()
        self.local_extractor.eval()

        with torch.no_grad():
            global_feat = self.global_extractor(input_tensor)  
            features, feats = self.local_extractor(input_tensor, global_feat)  # (B, 256, 32, 32)

            # fastflow_features_path = "/home/park/Documents/ws_yspark/toss/txt/fastflow_features.txt"  
            # features_temp = features.squeeze(0).detach().cpu() 
            # c, h, w = features_temp.shape

            # with open(fastflow_features_path, 'w') as f:
            #     f.write(f"Shape: ({c}, {h}, {w})\n")

            #     for ch in range(c):
            #         f.write("[\n")
            #         for row in range(h):
            #             row_values = ", ".join(f"{features_temp[ch, row, col].item():.6f}" for col in range(w))
            #             f.write(f"  [{row_values}]\n")
            #         f.write("]\n\n")

            norm_features = F.normalize(features, p=2, dim=1)

            # fastflow_norm_features_path = "/home/park/Documents/ws_yspark/toss/txt/fastflow_norm_features.txt"  
            # norm_features_temp = norm_features.squeeze(0).detach().cpu() 
            # c, h, w = norm_features_temp.shape

            # with open(fastflow_norm_features_path, 'w') as f:
            #     f.write(f"Shape: ({c}, {h}, {w})\n")

            #     for ch in range(c):
            #         f.write("[\n")
            #         for row in range(h):
            #             row_values = ", ".join(f"{norm_features_temp[ch, row, col].item():.6f}" for col in range(w))
            #             f.write(f"  [{row_values}]\n")
            #         f.write("]\n\n")

            output, _ = self.nf_flows(norm_features)  # (B, 256, 32, 32)
            
            # fastflow_output_path = "/home/park/Documents/ws_yspark/toss/txt/fastflow_output.txt"  
            # output_temp = output.squeeze(0).detach().cpu() 
            # c, h, w = output_temp.shape

            # with open(fastflow_output_path, 'w') as f:
            #     f.write(f"Shape: ({c}, {h}, {w})\n")

            #     for ch in range(c):
            #         f.write("[\n")
            #         for row in range(h):
            #             row_values = ", ".join(f"{output_temp[ch, row, col].item():.6f}" for col in range(w))
            #             f.write(f"  [{row_values}]\n")
            #         f.write("]\n\n")

            inter_output = F.interpolate(output, self.input_size, mode='bilinear', align_corners=False)  # (B, 256, 512, 512)

            fastflow_inter_output_path = "/home/park/Documents/ws_yspark/toss/txt/fastflow_inter_output.txt"  
            inter_output_temp = inter_output.squeeze(0).detach().cpu() 
            c, h, w = inter_output_temp.shape

            with open(fastflow_inter_output_path, 'w') as f:
                f.write(f"Shape: ({c}, {h}, {w})\n")

                for ch in range(c):
                    f.write("[\n")
                    for row in range(h):
                        row_values = ", ".join(f"{inter_output_temp[ch, row, col].item():.6f}" for col in range(w))
                        f.write(f"  [{row_values}]\n")
                    f.write("]\n\n")

            inter_output = inter_output.squeeze(0)  # (256, 512, 512)
            d, h, w = inter_output.shape
            output_flat = inter_output.view(d, -1)  # (256, 512*512)
            similarity_map = proxy.inference(output_flat)  # (512*512)
            similarity_map = similarity_map.view(1, 1, h, w).squeeze(0)  # (1, 1,512, 512)

            return similarity_map