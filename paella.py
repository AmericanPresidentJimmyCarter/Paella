import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch import nn, optim
import torchvision
from tqdm import tqdm
import time
import numpy as np
from modules import DenoiseUNet
import requests
from utils import (
    sample,
    encode,
    decode,
    b64_string_to_tensor,
)

from rudalle import get_vae
from ema import ModelEma
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

URL_BATCH = 'http://127.0.0.1:4456/batch'
URL_CONDITIONING = 'http://127.0.0.1:4455/conditioning'


def train(args):
    if os.path.exists(f"models/{args.run_name}/pytorch_model.bin"):
        resume = True
    else:
        resume = False
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        accelerator.print(f"Starting run '{args.run_name}'....")
        accelerator.print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")

    vqmodel = get_vae(cache_dir=args.cache_dir).to(device)
    vqmodel.eval().requires_grad_(False)

    # wait for everyone to load vae
    accelerator.wait_for_everyone()

    model = DenoiseUNet(num_labels=args.num_codebook_vectors, c_clip=2048).to(device)
    # wait for everyone to load model
    accelerator.wait_for_everyone()

    accelerator.print(
        f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}"
    )

    lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = nn.MSELoss()

    if accelerator.is_main_process:
        wandb.watch(model)
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        total_steps=args.total_steps,
        max_lr=lr,
        pct_start=0.1 if not args.finetune else 0.0,
        div_factor=25,
        final_div_factor=1 / 25,
        anneal_strategy="linear",
    )

    dataset = None
    model, optimizer, dataset, scheduler = accelerator.prepare(model, optimizer,
        dataset, scheduler)

    if resume:
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0
        accelerator.print("Loading last checkpoint....")
        # logs = torch.load(f"results/{args.run_name}/log.pt")
        # start_step = logs["step"] + 1
        # losses = logs["losses"]
        # accuracies = logs["accuracies"]
        # total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        # model.load_state_dict(
        #     torch.load(f"models/{args.run_name}/model.pt", map_location=device)
        # )
        # accelerator.print("Loaded model.")
        # opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        # last_lr = opt_state["param_groups"][0]["lr"]
        # with torch.no_grad():
        #     for _ in range(logs["step"]):
        #         scheduler.step()
        # accelerator.print(f"Initialized scheduler")
        # accelerator.print(
        #     f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}"
        # )
        # optimizer.load_state_dict(opt_state)
        # del opt_state
        accelerator.load_state(f"models/{args.run_name}/")

    else:
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0

    model_ema = None
    if args.ema:
        model_ema = ModelEma(
            model=model,
            decay=args.ema_decay,
            device='cpu',
            ema_model_path=args.ema_model_path,
        )

    accelerator.wait_for_everyone()

    pbar = tqdm(
        total=args.total_steps,
        initial=start_step,
    ) if accelerator.is_main_process else None
    # should we prepare vqmodel, clip_model, t5_model?

    model.train()
    step = 0
    epoch = 0

    while step < args.total_steps:
        resp_dict = None
        try:
            resp = requests.post(url=URL_BATCH, timeout=5)
            resp_dict = resp.json()
        except Exception:
            import traceback
            traceback.print_exc()
            continue

        if 'images' not in resp_dict or resp_dict['images'] is None or \
            'captions' not in resp_dict or resp_dict['captions'] is None or \
            'conditioning_flat' not in resp_dict or resp_dict['conditioning_flat'] is None or \
            'conditioning_full' not in resp_dict or resp_dict['conditioning_full'] is None or \
            'unconditioning_flat' not in resp_dict or resp_dict['unconditioning_flat'] is None or \
            'unconditioning_full' not in resp_dict or resp_dict['unconditioning_full'] is None:
            continue

        images = b64_string_to_tensor(resp_dict['images'], device)
        captions = resp_dict['captions']
        text_embeddings = b64_string_to_tensor(resp_dict['conditioning_flat'],
            device)
        text_embeddings_full = b64_string_to_tensor(resp_dict['conditioning_full'],
            device)
        text_embeddings_uncond = b64_string_to_tensor(resp_dict['unconditioning_flat'],
            device)
        text_embeddings_full_uncond = b64_string_to_tensor(resp_dict['unconditioning_full'],
            device)
        if text_embeddings is None or text_embeddings_full is None or \
            text_embeddings_uncond is None or text_embeddings_full_uncond is None:
            continue

        image_indices = encode(vqmodel, images)

        r = torch.rand(images.size(0), device=device)
        noised_indices, mask = model.module.add_noise(image_indices, r)

        if (
            np.random.rand() < 0.1
        ):  # 10% of the times -> unconditional training for classifier-free-guidance
            # Old method:
            # text_embeddings = images.new_zeros(images.size(0), 2048)
            # text_embeddings_full = images.new_zeros(images.size(0), 77, 2048)
            # New method:
            text_embeddings = text_embeddings_uncond
            text_embeddings_full = text_embeddings_full_uncond

        pred = model(noised_indices, text_embeddings, r, text_embeddings_full)
        image_indices = image_indices.to(device)
        loss = criterion(pred, image_indices)
        loss_adjusted = loss / args.accum_grad

        accelerator.backward(loss_adjusted)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        acc = (pred.argmax(1) == image_indices).float()
        acc = acc.mean()

        total_loss += loss.item()
        total_acc += acc.item()
        if accelerator.is_main_process:
            log = {
                "loss": total_loss / (step + 1),
                "acc": total_acc / (step + 1),
                "curr_loss": loss.item(),
                "curr_acc": acc.item(),
                "ppx": np.exp(total_loss / (step + 1)),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
            }
            pbar.set_postfix(log)
            wandb.log(log)

        if (
            model_ema is not None
            and accelerator.is_main_process
            and step % args.ema_update_steps == 0
        ):
            accelerator.print(f"EMA weights are being updated and saved ({step=})")
            model_ema.update(model)
            torch.save(model_ema.module, args.ema_model_path)

        # All of this is only done on the main process
        if step % args.log_period == 0 and accelerator.is_main_process:
            accelerator.print(
                f"Step {step} - loss {total_loss / (step + 1)} - acc {total_acc / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}"
            )

            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                n = 1
                images = images[: args.comparison_samples]
                image_indices = image_indices[: args.comparison_samples]
                captions = captions[: args.comparison_samples]
                text_embeddings = text_embeddings[: args.comparison_samples]
                text_embeddings_full = text_embeddings_full[: args.comparison_samples]
                sampled = sample(
                    model, c=text_embeddings, c_full=text_embeddings_full,
                    c_uncond=text_embeddings_uncond[: args.comparison_samples],
                    c_full_uncond=text_embeddings_full_uncond[: args.comparison_samples],
                )  # [-1]
                sampled = decode(vqmodel, sampled)
                recon_images = decode(vqmodel, image_indices)

                if args.log_captions:
                    # cool_captions_data = torch.load("cool_captions.pth")
                    # cool_captions_text = cool_captions_data["captions"]
                    cool_captions_text = args.cool_captions_text

                    resp_dict = None
                    try:
                        resp = requests.post(url='http://127.0.0.1:4455/conditionings', json={
                            'captions': cool_captions_text,
                        }, timeout=5)
                        resp_dict = resp.json()
                    except Exception:
                        import traceback
                        traceback.print_exc()
                    cool_captions_embeddings = b64_string_to_tensor(resp_dict['flat'],
                        device)
                    cool_captions_embeddings_full = b64_string_to_tensor(resp_dict['full'],
                        device)

                    # cool_captions_embeddings = generate_clip_embeddings(clip_model,
                    #     text_tokens)

                    cool_captions = DataLoader(
                        TensorDataset(
                            cool_captions_embeddings.repeat_interleave(n, dim=0)
                        ),
                        batch_size=len(args.cool_captions_text),
                    )
                    cool_captions_sampled = []
                    st = time.time()
                    for caption_embedding in cool_captions:
                        caption_embedding = caption_embedding[0].float().to(device)
                        sampled_text = sample(
                            model,
                            c=caption_embedding,
                            c_full=cool_captions_embeddings_full,
                            c_uncond=text_embeddings_uncond[: len(cool_captions_text)],
                            c_full_uncond=text_embeddings_full_uncond[: len(cool_captions_text)],
                        )  # [-1]
                        sampled_text = decode(vqmodel, sampled_text)
                        # sampled_text_ema = decode(vqmodel, sampled_text_ema)
                        for s in sampled_text:
                            cool_captions_sampled.append(s.cpu())
                            # cool_captions_sampled_ema.append(t.cpu())
                    accelerator.print(
                        f"Took {time.time() - st} seconds to sample {len(cool_captions_text) * 2} captions."
                    )

                    cool_captions_sampled = torch.stack(cool_captions_sampled)
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(cool_captions_sampled, nrow=11),
                        os.path.join(
                            f"results/{args.run_name}", f"cool_captions_{step:03d}.png"
                        ),
                    )

                    # cool_captions_sampled_ema = torch.stack(cool_captions_sampled_ema)
                    # torchvision.utils.save_image(
                    #     torchvision.utils.make_grid(cool_captions_sampled_ema, nrow=11),
                    #     os.path.join(f"results/{args.run_name}", f"cool_captions_{step:03d}_ema.png")
                    # )

                log_images = torch.cat(
                    [
                        torch.cat([i for i in sampled.cpu()], dim=-1),
                    ],
                    dim=-2,
                )

            model.train()

            torchvision.utils.save_image(
                log_images, os.path.join(f"results/{args.run_name}", f"{step:03d}.png")
            )

            log_data = [
                [captions[i]]
                + [wandb.Image(sampled[i])]
                + [wandb.Image(images[i])]
                + [wandb.Image(recon_images[i])]
                for i in range(len(captions))
            ]
            log_table = wandb.Table(
                data=log_data, columns=["Caption", "Image", "Orig", "Recon"]
            )
            wandb.log({"Log": log_table})

            if args.log_captions:
                log_data_cool = [
                    [cool_captions_text[i]] + [wandb.Image(cool_captions_sampled[i])]
                    for i in range(len(cool_captions_text))
                ]
                log_table_cool = wandb.Table(
                    data=log_data_cool, columns=["Caption", "Image"]
                )
                wandb.log({"Log Cool": log_table_cool})
                del sampled_text, log_data_cool

            del sampled, log_data

            if step % args.extra_ckpt == 0:
                # torch.save(
                #     model.state_dict(), f"models/{args.run_name}/model_{step}.pt"
                # )
                # torch.save(
                #     optimizer.state_dict(),
                #     f"models/{args.run_name}/model_{step}_optim.pt",
                # )
                accelerator.save_state(f"models/{args.run_name}/{step}/")

            # torch.save(model.state_dict(), f"models/{args.run_name}/model.pt")
            # torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            # torch.save(
            #     {"step": step, "losses": losses, "accuracies": accuracies},
            #     f"results/{args.run_name}/log.pt",
            # )
            if step % args.write_every_step == 0:
                accelerator.save_state(f"models/{args.run_name}/")

        del images, image_indices, r, text_embeddings
        del noised_indices, mask, pred, loss, loss_adjusted, acc

        if accelerator.is_main_process:
            # This is the main process only, so increment by the number of
            # devices.
            pbar.update(len(args.devices))
            step += len(args.devices)

    accelerator.print(f"Training complete (steps: {step}, epochs: {epoch})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "paella-0"
    args.model = "UNet"
    args.dataset_type = "webdataset"
    args.total_steps = 10_000_000
    # Be sure to sync with TARGET_SIZE in utils.py and condserver/data.py
    args.image_size = 256
    args.log_period = 2_500
    args.extra_ckpt = 100_000
    args.write_every_step = 1_000
    args.ema = True
    args.ema_decay = 0.9999
    args.ema_update_steps = 500_000
    args.ema_model_path = "ema_weights.ckpt"
    args.accum_grad = 1
    args.num_codebook_vectors = 8192
    args.log_captions = True
    args.finetune = False
    args.comparison_samples = 5
    args.cool_captions_text = [
        "a cat is sleeping",
        "a painting of a clown",
        "a horse",
        "a river bank at sunset",
        "bon jovi playing a sold out show in egypt. you can see the great pyramids in the background",
        # "The citizens of Rome rebel against the patricians, believing them to be hoarding all of the food and leaving the rest of the city to starve",
        # "King Henry rouses his small, weak, and ill troops, telling them that the less men there are, the more honour they will all receive.",
        # "Upon its outward marges under the westward mountains Mordor was a dying land, but it was not yet dead. And here things still grew, harsh, twisted, bitter, struggling for life.",
    ]
    parallel_init_dir = "/data"
    args.parallel_init_file = f"file://{parallel_init_dir}/dist_file"
    args.wandb_project = "project"
    args.wandb_entity = "entity"
    # args.cache_dir = "/data/cache"  # cache_dir for models
    args.cache_dir = "/home/user/.cache"
    args.offload = False

    # Testing:
    # args.dataset_path = '/home/user/Programs/Paella/models/6.tar'
    # args.dataset_path = "gigant/oldbookillustrations_2"
    args.dataset_path = "laion/laion-coco"
    accelerator.print("Launching with args: ", args)
    train(args)
