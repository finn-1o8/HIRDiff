import argparse
import os
import json
import torch
import torch as th
from collections import OrderedDict
from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
from guided_diffusion.data import load_data
from utility import seed_everywhere, MSIQA
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the JSON config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        opt = json.load(f)
    opt = utils.dict_to_nonedict(opt)

    device = th.device("cuda" if opt.inference.params.gpu_ids != "-1" and torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.inference.params.gpu_ids
    print(f'INFO: Using device: {device}')

    model, diffusion = create_model_and_diffusion_RS(opt)

    gen_path = os.path.join(opt.inference.path.resume_state + "_gen.pth")
    if os.path.isabs(gen_path) is False:
        gen_path = os.path.join(opt.path.root if hasattr(opt, 'path') else '.', gen_path)
    print(f"INFO: Loading checkpoint: {gen_path}")
    cks = th.load(gen_path, map_location='cpu'); new_cks = OrderedDict()
    for k, v in cks.items():
        new_cks[k.replace('denoise_fn.', '')] = v

    if model.in_channels == 6 and new_cks['init_conv.weight'].shape[1] == 3:
        print("INFO: Inflating 3ch checkpoint weights for 6ch model.")
        w_rgb = new_cks['init_conv.weight']; new_w = torch.zeros_like(model.init_conv.weight.data)
        new_w[:, 0:3, :, :] = w_rgb; avg = torch.mean(w_rgb, dim=1, keepdim=True)
        for i in range(3, 6):
            new_w[:, i:i+1, :, :] = avg
        new_cks['init_conv.weight'] = new_w
        w_rgb_out = new_cks['final_conv.block.3.weight']; new_w_out = torch.zeros_like(model.final_conv.block[3].weight.data)
        new_w_out[0:3, :, :, :] = w_rgb_out; avg_out = torch.mean(w_rgb_out, dim=0, keepdim=True)
        for i in range(3, 6):
            new_w_out[i:i+1, :, :, :] = avg_out
        new_cks['final_conv.block.3.weight'] = new_w_out
        b_rgb_out = new_cks['final_conv.block.3.bias']; new_b_out = torch.zeros_like(model.final_conv.block[3].bias.data)
        new_b_out[0:3] = b_rgb_out; avg_b = torch.mean(b_rgb_out)
        for i in range(3, 6):
            new_b_out[i] = avg_b
        new_cks['final_conv.block.3.bias'] = new_b_out

    model.load_state_dict(new_cks, strict=False)
    model.to(device)
    model.eval()
    print("INFO: Model loaded and ready.")

    data_loader = load_data(opt)

    output_dir = opt.inference.path.results_root
    os.makedirs(output_dir, exist_ok=True)

    for data_batch in tqdm(data_loader, desc="Restoring Images"):
        if data_batch is None:
            continue
        input_tensor = data_batch['LQ'].to(device)
        gt_tensor = data_batch['GT'].to(device)
        current_path = data_batch['path'] if isinstance(data_batch['path'], str) else data_batch['path'][0]

        model_condition = {'input': input_tensor, 'gt': gt_tensor}
        try:
            sigma_val = float(opt.inference.params.task_params) / 255.0
            model_condition['sigma'] = torch.full_like(input_tensor[:,0:1,...], sigma_val).float().to(device)
        except Exception:
            pass

        Ch = model.in_channels
        param = {
            'task': opt.inference.params.get('task', 'denoise'),
            'eta1': opt.inference.params.eta1,
            'eta2': opt.inference.params.eta2,
            'k': opt.inference.params.k
        }
        param['Band'] = torch.arange(Ch, device=device).long()

        sample_out, _ = diffusion.p_sample_loop(
            model,
            (1, Ch, opt.dataset.image_size, opt.dataset.image_size),
            Rr=Ch,
            step=opt.inference.params.step,
            clip_denoised=True,
            model_condition=model_condition,
            param=param,
            progress=False
        )

        final_im_out = torch.clip((sample_out + 1) / 2, 0, 1)

        psnr, ssim = MSIQA(final_im_out, gt_tensor) if gt_tensor is not None else (-1.0, -1.0)
        print(f"  Processed: {os.path.basename(str(current_path))}, PSNR:{psnr:.2f}, SSIM:{ssim:.4f}")

        out_fname = os.path.basename(str(current_path)).replace(".npy", f"_restored_s{opt.inference.params.step}.npy")
        np.save(os.path.join(output_dir, out_fname), final_im_out.squeeze(0).cpu().numpy())

    print(f"\u2705\u2705\u2705 DONE! Restored images saved to: {output_dir}")
