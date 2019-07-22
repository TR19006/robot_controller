import math
from collections import OrderedDict
import torch

def reglayer_scale(size, num_layer, size_the):
    reg_layer_size = []
    for i in range(num_layer + 1):
        size = math.ceil(size / 2.)
        if i >= 2:
            reg_layer_size += [size]
            if i == num_layer and size_the != 0:
                reg_layer_size += [size - size_the]
    return reg_layer_size

def get_scales(size, size_pattern):
    size_list = []
    for x in size_pattern:
        size_list += [round(x * size, 2)]
    return  size_list

def aspect_ratio(num):
    as_ra = []
    for _ in range(num):
        as_ra += [[2, 3]]
    return as_ra

def mk_anchors(size, multiscale_size, size_pattern, step_pattern, num_reglayer = 6, param = 2):
    cfg = dict()
    cfg['feature_maps'] = reglayer_scale(size, num_reglayer, param if size >= multiscale_size else 0)
    cfg['min_dim'] = size
    cfg['steps'] = step_pattern
    cfg['min_sizes'] = get_scales(multiscale_size, size_pattern[:-1])
    cfg['max_sizes'] = get_scales(multiscale_size, size_pattern[1:])
    cfg['aspect_ratios'] = aspect_ratio(num_reglayer)
    cfg['variance'] = [0.1, 0.2]
    cfg['clip'] = True
    return cfg

def anchors(cfg):
    return mk_anchors(cfg.model.input_size,
                      cfg.model.input_size,
                      cfg.model.anchor_config.size_pattern, 
                      cfg.model.anchor_config.step_pattern)
    
def init_net(net, cfg, resume_net, cuda):
    if cfg.model.init_net and not resume_net:
        net.init_model(cfg.model.pretrained)
    else:
        if cuda:
            state_dict = torch.load(resume_net)
        else:
            state_dict = torch.load(resume_net, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict, strict=False)

def to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
