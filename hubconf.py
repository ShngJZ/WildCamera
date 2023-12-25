dependencies = ['torch']
import torch
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

def WildCamera(pretrained=True):
    model = NEWCRFIF(version='large07', pretrained=None)
    if pretrained:
        pretrained_resource = "https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/checkpoint/wild_camera_all.pth"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == "__main__":
    model_zoe_n = torch.hub.load('ShngJZ/WildCamera', "WildCamera", pretrained=True)