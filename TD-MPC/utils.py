from collections import deque

import cv2
import torch as T

def symlog(x):
    # return x
    return T.sign(x) * T.log(1 + T.abs(x))

def symexp(x):
    return T.sign(x) * (T.exp(T.abs(x)) - 1)

def make_frame_stacker(k):
    frames = deque(maxlen=k)
    def reset(frame):          # frame: (3, 64, 64)
        for _ in range(k):
            frames.append(frame)
        return T.cat(list(frames), dim=0)   # (3k, 64, 64)
    def push(frame):
        frames.append(frame)
        return T.cat(list(frames), dim=0)
    return reset, push

def update_target_network(epoch, value_predictor, offline_value_predictor, tau=0.005):
    """
    Update the offline value predictor using a slow-moving average of the online value predictor.
    """
    with T.no_grad():
        for target_param, param in zip(offline_value_predictor.parameters(), value_predictor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def process_image(image):
    # Resize to 64x64. Keep as uint8 [0, 255] -- normalization to [-1, 1] happens
    # inside RepresentationModel.forward so the replay buffer can store uint8.
    image = cv2.resize(image, (64, 64))

    # HWC -> CHW, still uint8
    image = T.from_numpy(image).permute(2, 0, 1).contiguous()

    return image

def random_shift(imgs, pad=4):
    """
    DrQ-style image augmentation: replicate-pad by `pad` pixels on every side,
    then take a random crop back to the original H x W. One random shift per
    batch element, shared across all channels / stacked frames.

    imgs: (B, C, H, W), any dtype (H == W). Returns float32, same shape.
    Applied ONLY during the gradient update -- never during planning.
    """
    imgs = imgs.float()
    b, c, h, w = imgs.shape
    imgs = T.nn.functional.pad(imgs, (pad, pad, pad, pad), mode="replicate")

    # normalized sampling grid for the un-shifted crop
    eps = 1.0 / (h + 2 * pad)
    arange = T.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad,
                        device=imgs.device, dtype=imgs.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = T.cat([arange, arange.transpose(1, 0)], dim=2)   # (h, w, 2)
    base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)        # (b, h, w, 2)

    # random integer pixel shift in [0, 2*pad], expressed in grid units
    shift = T.randint(0, 2 * pad + 1, size=(b, 1, 1, 2),
                      device=imgs.device, dtype=imgs.dtype)
    shift *= 2.0 / (h + 2 * pad)

    grid = base_grid + shift
    return T.nn.functional.grid_sample(imgs, grid, padding_mode="zeros",
                                       align_corners=False)