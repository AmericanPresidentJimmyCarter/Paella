import torch
import torchvision
from torch.utils.data import DataLoader
from random import randrange
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
import math
from einops import rearrange

TARGET_SIZE = 256

def resize_image(img):
    width, height = img.size   # Get dimensions

    rz_w = width
    rz_h = height
    _m = max(width, height)
    if _m == width:
        rz_h = TARGET_SIZE
        rz_w = math.floor((rz_w / rz_h) * TARGET_SIZE)
    if _m == height:
        rz_h = math.floor((rz_h / rz_w) * TARGET_SIZE)
        rz_w = TARGET_SIZE
    
    img = img.resize((rz_w, rz_h), resample=PIL.Image.LANCZOS)

    return img


def crop_random(img):
    img_size = img.size
    x_max = img_size[0] - TARGET_SIZE
    y_max = img_size[1] - TARGET_SIZE

    random_x = randrange(0, x_max//2 + 1) * 2
    random_y = randrange(0, y_max//2 + 1) * 2

    area = (random_x, random_y, random_x + TARGET_SIZE, random_y + TARGET_SIZE)
    c_img = img.crop(area)
    return c_img


def encode(vq, x):
    return vq.model.encode((2 * x - 1))[-1][-1]


def decode(vq, z):
    return vq.decode(z.view(z.shape[0], -1))


def arr_to_pil(img_arrs):
    images = []
    for img_arr in img_arrs:
        x_sample_c = 255. * rearrange(img_arr.cpu().numpy(),
            'c h w -> h w c')
        img = Image.fromarray(x_sample_c.astype(np.uint8))
        buffered = BytesIO()
        img.save(buffered, format='PNG')

        images.append(img)
    return images


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def sample(
    model,
    c,
    x=None,
    mask=None,
    T=12,
    size=(TARGET_SIZE // 8, TARGET_SIZE // 8),
    starting_t=0,
    temp_range=[1.0, 1.0],
    typical_filtering=True,
    typical_mass=0.2,
    typical_min_tokens=1,
    classifier_free_scale=-1,
    renoise_steps=11,
    renoise_mode='start',
    c_full=None,
    c_uncond=None,
    c_full_uncond=None,
):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        elif mask is not None:
            noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            x = noise * mask + (1-mask) * x
        init_x = x.clone()
        for i in range(starting_t, T):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]

            lfcu = []
            if classifier_free_scale >= 0 and \
                c_uncond is not None and \
                c_full_uncond is not None:
                # TODO Remove when this is fixed
                c_uncond = torch.zeros_like(c)
                c_full_uncond = torch.zeros_like(c_full)

                logits_from_c_uncond_00 = model(x, c_uncond, r, c_full_uncond)
                logits_from_c_uncond_10 = model(x, c, r, c_full_uncond)
                logits_from_c_uncond_01 = model(x, c_uncond, r, c_full)
                lfcu = [logits_from_c_uncond_00, logits_from_c_uncond_10,
                    logits_from_c_uncond_01]

            logits = model(x, c, r, c_full)

            if classifier_free_scale >= 0 and len(lfcu) == 0:
                print('Warning: you are sampling with classifier free guidance ' +
                    'but you have not provided unconditioned embeddings. ' +
                    'Please provide c_uncond and c_full_uncond to this ' +
                    'function.')
            if classifier_free_scale >= 0 and len(lfcu) > 0:
                # logits_uncond = model(x, torch.zeros_like(c), r, torch.zeros_like(c_full))
                # logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
                logits_00 = logits - lfcu[0]
                logits_10 = logits - lfcu[1]
                logits_01 = logits - lfcu[2]

                logits_delta = torch.sum(torch.stack(
                    [logits_00, logits_10, logits_01]), 0)
                logits = lfcu[0] + (logits_delta * classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))

            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            # x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            x_flat = gumbel_sample(x_flat, temperature=temp)
            x = x_flat.view(x.size(0), *x.shape[2:])
            if mask is not None:
                x = x * mask + (1-mask) * init_x
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                else:  # 'rand'
                    x, _ = model.add_noise(x, r_range[i+1])
    return x.detach()


class ProcessData:
    def __init__(self, image_size=256):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.RandomCrop(image_size),
        ])

    def __call__(self, data):
        data["jpg"] = self.transforms(data["jpg"])
        return data


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def collate(batch):
    images = torch.cat([preprocess(crop_random(resize_image(i['1600px']))) for i in batch], 0)
    captions = [i['image_alt'] if i.get('image_alt', None) is not None else
        i.get('image_caption', '') for i in batch]
    return [images, captions]


def get_dataloader(args):
    import datasets
    dataset = datasets.load_dataset("gigant/oldbookillustrations_2", split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=collate)
    return dataloader
