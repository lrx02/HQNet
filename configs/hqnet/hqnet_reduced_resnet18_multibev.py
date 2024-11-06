work_dirs = "work_dirs/hqnet/reduced_r18_multibev"
detection_pattern = 'det2d'
dataset_type = 'MultiBEV'
dataset_type_cam = 'MultiBEV4CAM'
dataset_type_sfm = 'MultiBEV4SFM'
test_json_file = '../TuSimple/LaneDetection/test_label.json'
cache_dir = "./cache"

ori_img_w = 800
ori_img_h = 800
img_norm = dict(mean=[0., 0., 0.], std=[1., 1., 1.])
img_h = 640
img_w = 640
cut_height = 0
img_res = 0.03128911138923655

max_points = 80
max_lanes = 9

attn_dim = 128
dim_feedforward = 128
num_enc = 3
num_dec = 3

num_queries = 15
sampling_steps = 40
roi_r = 18
queue_length = 3
batch_size = 8*queue_length
overlap_scale = 2

net = dict(type='Detector', )

# CAM
backbone = dict(
    type='ResNetWrapper_Reduced',
    resnet='resnet18',
    pretrained=False,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

positional_encoding = dict(
    type='SinePositionalEncoding',
    num_pos_feats=attn_dim,
    normalize=True
)

transformer = dict(
    type='DETR',
    d_model=attn_dim,
    dropout=0.1,
    nhead=4,
    dim_feedforward=dim_feedforward,
    num_encoder_layers=num_enc,
    num_decoder_layers=num_dec,
    normalize_before=False,
    return_intermediate_dec=True
)

cam_head = dict(
    type='CAMHead',
)

# SFM
box_positional_encoding = dict(
    type='BoxPositionalEncoding',
    num_pos_feats=attn_dim,
    normalize=True)

fusion_transformer =  dict(
    type='Fusion_Transformer',
    d_model=attn_dim,
    dropout=0.1,
    nhead=4,
    dim_feedforward=dim_feedforward,
    num_encoder_layers=num_enc,
    num_decoder_layers=num_dec,
    normalize_before=False,
    return_intermediate_dec=True
)


sfm_head = dict(
    type='SFMHead'
)

loss_ce_weight = 1.
loss_curves_weight = 1.6667
loss_lowers_weight = 2.5
loss_uppers_weight = 2.5

matcher = dict(
    type='HungarianMatcher',
    cost_class=loss_ce_weight,
    curves_weight=loss_curves_weight,
    lower_weight=loss_lowers_weight,
    upper_weight=loss_uppers_weight
)

losses = ['labels', 'curves']
# sfm_losses = ['position'] # ['position', 'shape']
loss_loc_weight = 1.
loss_shape_weight = 0.1


train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='VerticalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-15, 15),
                                 scale=(0.8, 1.2)),
                 p=0.9),
            # dict(name='CropToFixedSize', parameters=dict(height=775, width=775), p=1.0),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'track', 'seg']),
]


train_sfm_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'track', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img', 'lane_line', 'track', 'seg']),
]

dataset = dict(train=dict(
    type=dataset_type_cam,
    split='train',
    processes=train_process,
),
train_sfm=dict(
    type=dataset_type_sfm,
    split='train',
    processes=train_sfm_process,
),
val=dict(
    type=dataset_type_sfm,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type_sfm,
    split='test',
    processes=val_process,
))

epochs = 200
cam_epoch = 10

total_iter = (16159 // batch_size) * cam_epoch + (16159 // (batch_size // queue_length)) * (epochs - cam_epoch)
optimizer = dict(type='AdamW', lr=2.0e-4, weight_decay=1.0e-2)
# scheduler = dict(type='CosineAnnealingLR', T_max=epochs, eta_min=(optimizer['lr'] * 1.0e-3))
scheduler = dict(type='WarmupCosineAnnealingLR', warmup_steps=1000, T_max=total_iter, warmup_factor=1.0/3, eta_min=(optimizer['lr'] * 1.0e-3))

eval_ep = 1
save_ep = 1
workers = 8
# workers = 0
log_interval = 100
lr_update_by_epoch = False
