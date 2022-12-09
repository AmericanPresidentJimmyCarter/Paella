import torch

from modules import DenoiseUNet
from condserver.t5 import FrozenT5Embedder
import open_clip
from open_clip import tokenizer
from rudalle import get_vae

from condserver.data import arr_to_pil, sample, decode


def generate_clip_embeddings(clip_model, clip_text_tokens) -> torch.Tensor:
    '''
    Get the CLIP embedding before feature extraction/normalization.
    TODO Alter the unet to use this instead of the final squished embedding.
    '''
    cast_dtype = clip_model.transformer.get_cast_dtype()

    x = clip_model.token_embedding(clip_text_tokens).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    return x


captions = [
    'a black cat is sleeping',
    # 'sunset over ibiza, palm trees are visible on the beach',
    # 'a brown shoe',
    # 'a blue t-shirt',
]

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k",
)
del clip_model.visual
clip_model = clip_model.to('cpu').eval().requires_grad_(False)
t5_model = FrozenT5Embedder(device='cpu').to('cpu')

model = DenoiseUNet(8192, c_clip=2048)
model.load_state_dict(
    torch.load('models/models/paellaaa-2/pytorch_model.bin', map_location='cuda')
)
model = model.to('cuda')

text_tokens = tokenizer.tokenize(captions)
text_tokens = text_tokens.to('cpu')
clip_embeddings = clip_model.encode_text(text_tokens).float().to('cuda')
clip_embeddings_full = generate_clip_embeddings(clip_model, text_tokens).float().to('cuda')
t5_embeddings_full = t5_model(captions).to('cuda')
text_embeddings = torch.cat([clip_embeddings, torch.mean(t5_embeddings_full, dim=1)], 1).to('cuda')
text_embeddings_full = torch.cat([clip_embeddings_full, t5_embeddings_full], 2).to('cuda')

text_tokens_uncond = tokenizer.tokenize([''] * len(captions))
text_tokens_uncond = text_tokens_uncond.to('cpu')
clip_embeddings_uncond = clip_model.encode_text(text_tokens_uncond).float().to('cuda')
clip_embeddings_full_uncond = generate_clip_embeddings(clip_model, text_tokens_uncond).float().to('cuda')
t5_embeddings_full_uncond = t5_model([''] * len(captions)).to('cuda')
text_embeddings_uncond = torch.cat([clip_embeddings_uncond, torch.mean(t5_embeddings_full_uncond, dim=1)], 1).to('cuda')
text_embeddings_full_uncond = torch.cat([clip_embeddings_full_uncond, t5_embeddings_full_uncond], 2).to('cuda')
del clip_model
del t5_model

sampled = sample(model,
    c=text_embeddings,
    c_full=text_embeddings_full,
    c_uncond=text_embeddings_uncond,
    c_full_uncond=text_embeddings_full_uncond,
    # typical_filtering=False,
    classifier_free_scale=1.5,
)
vqmodel = get_vae().to('cuda')
vqmodel.eval().requires_grad_(False)
sampled = decode(vqmodel, sampled.to('cuda'))
images  = arr_to_pil(sampled)
for i, image in enumerate(images):
    image.save(f'{i}.png')
