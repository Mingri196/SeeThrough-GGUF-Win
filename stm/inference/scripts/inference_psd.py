# __STG_PATCHED_INFERENCE_PSDPY__
import os.path as osp
import argparse
import sys
import os
import gc
from pathlib import Path
from PIL import Image

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')


import numpy as np
import torch

from utils.io_utils import find_all_imgs
from utils import inference_utils
from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr, resolve_target_parts
from utils.torch_utils import seed_everything
from annotators.animeinsseg.skytnt_segmentation import apply_segmentation, resolve_segmentation_resolution
from annotators.animeinsseg.animeseg_refine_model import load_refinenet


def save_masked_input(srcp, saved, mask_model=None, mask_resolution='auto'):
    with Image.open(srcp) as source:
        rgb = np.array(source.convert('RGB'))
        alpha_image = source.getchannel('A') if 'A' in source.getbands() else None
    if alpha_image is None or alpha_image.getextrema()[0] == 255:
        if mask_model is None:
            raise ValueError(f'{srcp} has no foreground alpha mask')
        mask_resolution = resolve_segmentation_resolution(rgb, mask_resolution)
        print(f'foreground mask resolution: {mask_resolution}')
        alpha = (apply_segmentation(rgb, model=mask_model, s=mask_resolution) >= 0.5).astype(np.uint8) * 255
    else:
        alpha = np.array(alpha_image)

    bbox = Image.fromarray(alpha).getbbox()
    if bbox is None:
        raise ValueError(f'empty foreground mask for {srcp}')

    saved = Path(saved)
    saved.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha).save(saved / 'input_mask.png')
    rgba = np.dstack((rgb, alpha))
    cropped = Image.fromarray(rgba).crop(bbox)
    cropped_path = saved / 'input_cropped.png'
    cropped.save(cropped_path)
    return str(cropped_path), bbox

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output')
    parser.add_argument('--srcp', type=str, default='assets/test_image.png', help='input image')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repo_id_layerdiff', default='layerdifforg/seethroughv0.0.2_layerdiff3d')
    parser.add_argument('--repo_id_depth', default='24yearsold/seethroughv0.0.1_marigold')
    parser.add_argument('--gguf_layerdiff', default=None, help='LayerDiff3D GGUF; overrides --repo_id_layerdiff')
    parser.add_argument('--gguf_depth', default=None, help='Marigold3D GGUF; overrides --repo_id_depth')
    parser.add_argument('--vae_ckpt', default=None)
    parser.add_argument('--unet_ckpt', default=None)
    parser.add_argument('--resolution', type=int, default=1280, help="inference resolution of layerdiff")
    parser.add_argument('--mask-resolution', default='auto',
                        help='foreground segmentation resolution (default: auto; or a positive integer)')
    parser.add_argument('--resolution_depth', default='auto',
                        help='depth inference resolution (default: auto; -1 matches LayerDiff; or a positive integer)')
    parser.add_argument('--inference_steps', type=int, default=30, help="inference steps of layerdiff")
    parser.add_argument('--inference_steps_depth', type=int, default=-1, help="inference steps of depth model")
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--save_cut_images', action='store_true', help='keep all extracted RGBA cut PNGs in the per-image output folder')
    parser.add_argument('--tblr_split', action='store_true', help='try split parts (handwear, eyes, etc) into left-right components')
    parser.add_argument('--use_lama', action=argparse.BooleanOptionalAction, default=True, help='use LaMa inpainting during post-processing (default on)')
    parser.add_argument('--ragdoll_split', action='store_true', help='split completed limbs into overlapping RTMPose-guided bone layers')
    parser.add_argument('--ragdoll-overlap', type=float, default=0.2,
                        help='joint overlap as a fraction of the shorter adjacent bone')
    parser.add_argument('--ragdoll-confidence', type=float, default=0.3,
                        help='minimum RTMPose joint confidence required for ragdoll split; '
                             'lower this for characters with occluded limbs (default: 0.3)')
    parser.add_argument('--disable_progressbar', action='store_true', help='hide progressbar')
    parser.add_argument('--group_offload', action='store_true')
    parser.add_argument('--quantized', action='store_true', help='keep GGUF weights quantized in VRAM, dequantize on the fly (Q4 stays Q4-sized)')
    parser.add_argument('--stream-weights', action=argparse.BooleanOptionalAction, default=None,
                        help='stream quantized weights from pinned RAM; default: auto by VRAM and resolution')
    parser.add_argument('--spatial-frame-chunk', type=int, default=None,
                        help='frames per spatial pass; default: auto, 0 disables chunking')
    parser.add_argument('--feed-forward-chunk', type=int, default=None,
                        help='transformer batch chunk; default: auto, 0 disables chunking')
    parser.add_argument('--temporal-token-chunk', type=int, default=None,
                        help='spatial tokens per temporal pass; default: auto, 0 disables chunking')
    parser.add_argument('--spill-frame-residuals', action=argparse.BooleanOptionalAction, default=None,
                        help='spill inactive UNet skips to CPU when estimated VRAM demand requires it')
    parser.add_argument('--attention-backend', choices=('sage', 'native', 'xformers'), default='sage',
                        help='Diffusers attention backend (default: SageAttention2)')
    parser.add_argument('--parts', nargs='+', default=None,
                        help='generate only these layers, for example: --parts face handwear "front hair"')
    args = parser.parse_args()
    if args.mask_resolution != 'auto':
        try:
            args.mask_resolution = int(args.mask_resolution)
            if args.mask_resolution <= 0:
                raise ValueError
        except ValueError:
            parser.error('--mask-resolution must be auto or a positive integer')
    if args.resolution_depth != 'auto':
        try:
            args.resolution_depth = int(args.resolution_depth)
            if args.resolution_depth == 0 or args.resolution_depth < -1:
                raise ValueError
        except ValueError:
            parser.error('--resolution_depth must be auto, -1, or a positive integer')
    try:
        resolve_target_parts(args.parts)
    except ValueError as error:
        parser.error(str(error))
    srcp = args.srcp
    layerdiff_model = args.gguf_layerdiff or args.repo_id_layerdiff
    depth_model = args.gguf_depth or args.repo_id_depth

    if osp.isdir(srcp):
        imglist = find_all_imgs(srcp, abs_path=True)
    else:
        imglist = [srcp]

    prepared = []
    mask_model = None
    for index, srcp in enumerate(imglist, 1):
        print(f'[{index}/{len(imglist)}] preparing {srcp}', flush=True)
        with Image.open(srcp) as source:
            has_mask = 'A' in source.getbands() and source.getchannel('A').getextrema()[0] < 255
        if not has_mask and mask_model is None:
            print('loading foreground segmenter...')
            mask_model = load_refinenet('animeseg')
        srcname = osp.basename(osp.splitext(srcp)[0])
        saved = osp.join(args.save_dir, srcname)
        cropped, bbox = save_masked_input(srcp, saved, mask_model, args.mask_resolution)
        print(f'cropped input saved to {cropped} (bbox={bbox})')
        prepared.append((srcp, cropped, srcname))

    del mask_model
    gc.collect()
    torch.cuda.empty_cache()

    for index, (srcp, cropped, srcname) in enumerate(prepared, 1):
        print(f'[{index}/{len(prepared)}] generating {srcname}', flush=True)

        seed_everything(args.seed)

        print('running LayerDiff...', flush=True)
        apply_layerdiff(cropped, layerdiff_model, save_dir=args.save_dir, seed=args.seed, vae_ckpt=args.vae_ckpt, unet_ckpt=args.unet_ckpt, \
            resolution=args.resolution, disable_progressbar=args.disable_progressbar, num_inference_steps=args.inference_steps, group_offload=args.group_offload,
            output_name=srcname, quantized=args.quantized, stream=args.stream_weights,
            spatial_frame_chunk=args.spatial_frame_chunk, temporal_token_chunk=args.temporal_token_chunk,
            feed_forward_chunk=args.feed_forward_chunk, spill_frame_residuals=args.spill_frame_residuals,
            attention_backend=args.attention_backend, target_tag_list=args.parts)

        print('running Marigold...', flush=True)
        apply_marigold(srcp, depth_model, save_dir=args.save_dir, seed=args.seed, disable_progressbar=args.disable_progressbar, \
            resolution=args.resolution_depth, num_inference_steps=args.inference_steps_depth, group_offload=args.group_offload,
            attention_backend=args.attention_backend)

        saved = osp.join(args.save_dir, srcname)
        further_extr(saved, rotate=False, save_to_psd=args.save_to_psd, tblr_split=args.tblr_split, use_lama=args.use_lama,
                     ragdoll_split=args.ragdoll_split, save_cut_images=args.save_cut_images,
                     ragdoll_overlap=args.ragdoll_overlap, ragdoll_confidence=args.ragdoll_confidence)
        if args.save_cut_images:
            print(f'cut images saved to {Path(saved) / "cuts"}')
