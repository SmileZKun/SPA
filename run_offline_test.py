import argparse
import logging
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from torchvision.transforms.functional import normalize

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results
from inference.LEDNet.basicsr.utils.registry import ARCH_REGISTRY
from inference.LEDNet.basicsr.utils.download_util import load_file_from_url
from inference.LEDNet.basicsr.utils import imwrite, img2tensor, tensor2img, scandir

logging.basicConfig(level=logging.INFO)


def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    # F:\ZHANGKUN/2/robotic-grasping-master\logs/230419_1029_/epoch_36_iou_0.97
    parser.add_argument('--network', type=str,
                        default="F:\ZHANGKUN/2/robotic-grasping-master\logs/230419_1029_/epoch_36_iou_0.97",
                        help='Path to saved network to evaluate')
    #F:\ZHANGKUN\cornell_dark\GT_gen_low\low1/pcd0130r.png  F:\ZHANGKUN\cornell_dark/real_dark\pcd0000r.png  F:\ZHANGKUN/2/robotic-grasping-master\hardware/realsense\pcd0000r.png
    parser.add_argument('--rgb_path', type=str, default='F:\ZHANGKUN\cornell_dark\ZK/pcd0055r.png',
                        help='RGB low light Image path')
    #F:\ZHANGKUN\cornell\data1/pcd0130d.tiff   F:\ZHANGKUN/2/robotic-grasping-master\hardware/realsense\pcd0000d.tiff
    parser.add_argument('--depth_path', type=str, default='F:\ZHANGKUN\cornell_dark\ZK/pcd0055d.tiff',
                        help='Depth Image path')
    parser.add_argument('--save_dir', type=str, default="F:\ZHANGKUN/2/robotic-grasping-master/results/real/4.21/22/",
                        help='Path to saved all images')

    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=1,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--model', type=str, default='lednet',
                        help='options: lednet, lednet_retrain, lednetgan')
    args = parser.parse_args()
    return args


pretrain_model_url = {
    'lednet': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
    'lednet_retrain': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
    'lednetgan': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednetgan.pth',
}

if __name__ == '__main__':

    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # ----------------  -- set up LEDNet network -------------------
    down_factor = 8  # check_image_size
    image_enhancement_net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False).to(device)

    # ckpt_path = 'weights/lednet.pth'
    assert args.model in ['lednet', 'lednet_retrain', 'lednetgan'], ('model name should be [lednet] or [lednetgan]')
    ckpt_path = load_file_from_url(url=pretrain_model_url[args.model],
                                   model_dir='F:\ZHANGKUN/2/robotic-grasping-master\inference\LEDNet\weights',
                                   progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params']
    image_enhancement_net.load_state_dict(checkpoint)
    image_enhancement_net.eval()
    # -------------------- start to processing ---------------------
    img = cv2.imread(args.rgb_path, cv2.IMREAD_COLOR)
    # prepare data
    img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    # without [-1,1] normalization in lednet model (paper version)
    if not args.model == 'lednet':
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0).to(device)

    # lednet inference
    with torch.no_grad():
        # check_image_size
        H, W = img_t.shape[2:]
        img_t = check_image_size(img_t, down_factor)
        output_t = image_enhancement_net(img_t)
        output_t = output_t[:, :, :H, :W]
        if args.model == 'lednet':
            output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
        else:
            output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        torch.cuda.empty_cache()

        output = output.astype('uint8')
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_pil = Image.fromarray(output_rgb)
        output_pil.save('F:\ZHANGKUN/2/robotic-grasping-master/result/output_pil.png')
        # # save restored img
        # save_restore_path = img_path.replace(args.test_path, result_root)
        # imwrite(output, save_restore_path)

    # Load image
    logging.info('Loading image...')
    pic = Image.open('F:\ZHANGKUN/2/robotic-grasping-master/result/output_pil.png')
    rgb = np.array(pic)
    rgb_image = Image.open(args.rgb_path, "r")
    rgb_low_light_show = np.array(rgb_image)
    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)

    logging.info('Done')

    img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
    start_time = time.time()
    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        avg_time = (time.time() - start_time)
        logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))
        if args.save:
            save_results(
                save_dir=args.save_dir,
                rgb_img=img_data.get_rgb(rgb_low_light_show, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                # rgb_enhance_img=img_data.get_rgb(rgb, False),
                rgb_enhance_img=None,
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb_low_light_show, False),
                         depth_img=np.squeeze(img_data.get_depth(depth)),
                         rgb_enhance_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img)
            fig.savefig('F:\ZHANGKUN/2/robotic-grasping-master/result/img_result.pdf')
