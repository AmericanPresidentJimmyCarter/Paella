import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch import nn, optim
import torchvision
from tqdm import tqdm
import time
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from t5 import FrozenT5Embedder
from modules import DenoiseUNet
from utils import get_dataloader, sample, encode, decode
import open_clip
from open_clip import tokenizer
from rudalle import get_vae
from ema import ModelEma


def generate_clip_embeddings(model, text_tokens) -> torch.Tensor:
    '''
    Get the CLIP embedding before feature extraction/normalization.

    TODO Alter the unet to use this instead of the final squished embedding.
    '''
    cast_dtype = model.transformer.get_cast_dtype()

    x = model.token_embedding(text_tokens).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    return x


def train(proc_id, args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True
    else:
        resume = False
    if not proc_id and args.node_id == 0:
        if resume:
            wandb.init(
                project="project",
                name=args.run_name,
                entity="your_entity",
                config=vars(args),
            )
        else:
            wandb.init(
                project="project",
                name=args.run_name,
                entity="your_entity",
                config=vars(args),
            )
        print(f"Starting run '{args.run_name}'....")
        print(
            f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}"
        )
    parallel = len(args.devices) > 1
    device = torch.device(proc_id)

    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(
            backend="nccl",
            init_method="file:///data/dist_file",
            world_size=args.n_nodes * len(args.devices),
            rank=proc_id + len(args.devices) * args.node_id,
        )
        torch.set_num_threads(6)

    model = DenoiseUNet(num_labels=args.num_codebook_vectors, c_clip=2048).to(device)

    if not proc_id and args.node_id == 0:
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s12b_b42k"
    )
    del clip_model.visual
    clip_model = clip_model.to(device).eval().half().requires_grad_(False)
    t5_model = FrozenT5Embedder(device=device).to(device)

    lr = 3e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if not proc_id and args.node_id == 0:
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

    if resume:
        if not proc_id and args.node_id == 0:
            print("Loading last checkpoint....")
        logs = torch.load(f"results/{args.run_name}/log.pt")
        start_step = logs["step"] + 1
        losses = logs["losses"]
        accuracies = logs["accuracies"]
        total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        model.load_state_dict(
            torch.load(f"models/{args.run_name}/model.pt", map_location=device)
        )
        if not proc_id and args.node_id == 0:
            print("Loaded model.")
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            for _ in range(logs["step"]):
                scheduler.step()
        if not proc_id and args.node_id == 0:
            print(f"Initialized scheduler")
            print(
                f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}"
            )
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0

    if parallel:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
        model = model.module

    model_ema = None
    if args.ema:
        model_ema = ModelEma(model=model, decay=args.ema_decay, device=device,
            ema_model_path=args.ema_model_path)

    # pbar = tqdm(
    #     enumerate(dataset, start=start_step),
    #     total=args.total_steps,
    #     initial=start_step) \
    #     if args.node_id == 0 and proc_id == 0 \
    #     else enumerate(dataset, start=start_step)
    model.train()
    # iterator = enumerate(dataset, start=start_step)
    pbar = tqdm(total=args.total_steps)

    batch_iterator = iter(dataset)
    step = 0
    while step < args.total_steps:
        try:
            images, captions = next(batch_iterator)
        except StopIteration:
            print("hit stop iteration")
            batch_iterator = iter(dataset)
            images, captions = next(batch_iterator)
        except Exception as e:
            import traceback

            traceback.print_exc()
            continue
        images = images.to(device)
        with torch.no_grad():
            image_indices = encode(vqmodel, images)
            r = torch.rand(images.size(0), device=device)
            noised_indices, mask = model.add_noise(image_indices, r)

            if (
                np.random.rand() < 0.1
            ):  # 10% of the times -> unconditional training for classifier-free-guidance
                text_embeddings = images.new_zeros(images.size(0), 2048)
                text_embeddings_full = images.new_zeros(images.size(0), 154, 1024)
                # text_embeddings = images.new_zeros(images.size(0), 77, 1024)
            else:
                text_tokens = tokenizer.tokenize(captions)
                text_tokens = text_tokens.to(device)
                clip_embeddings = clip_model.encode_text(text_tokens).float()
                clip_embeddings_full = generate_clip_embeddings(clip_model, text_tokens).float()
                t5_embeddings_full = t5_model(captions)
                text_embeddings = torch.cat([clip_embeddings, torch.mean(t5_embeddings_full, dim=1)], 1)
                text_embeddings_full = torch.cat([clip_embeddings_full, t5_embeddings_full], 1)

                # text_embeddings = generate_clip_embeddings(clip_model, text_tokens)

        pred = model(noised_indices, text_embeddings, r, text_embeddings_full)
        loss = criterion(pred, image_indices)
        loss_adjusted = loss / args.accum_grad


        loss_adjusted.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item()
        if (step + 1) % args.accum_grad == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        acc = (pred.argmax(1) == image_indices).float()
        acc = acc.mean()

        total_loss += loss.item()
        total_acc += acc.item()

        if not proc_id and args.node_id == 0:
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

        if model_ema is not None and args.node_id == 0 and proc_id == 0 and \
            step % args.ema_update_steps == 0:
            print(
                f"EMA weights are being updated and saved ({step=})"
            )
            model_ema.update(model)
            torch.save(model_ema.module, args.ema_model_path)

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:
            print(
                f"Step {step} - loss {total_loss / (step + 1)} - acc {total_acc / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}"
            )

            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                n = 1
                images = images[:args.comparison_samples]
                image_indices = image_indices[:args.comparison_samples]
                captions = captions[:args.comparison_samples]
                text_embeddings = text_embeddings[:args.comparison_samples]
                text_embeddings_full = text_embeddings_full[:args.comparison_samples]
                sampled = sample(model, c=text_embeddings,
                    c_full=text_embeddings_full)  # [-1]
                sampled = decode(vqmodel, sampled)
                recon_images = decode(vqmodel, image_indices)

                if args.log_captions:
                    # cool_captions_data = torch.load("cool_captions.pth")
                    # cool_captions_text = cool_captions_data["captions"]
                    cool_captions_text = args.cool_captions_text

                    text_tokens = tokenizer.tokenize(cool_captions_text)
                    text_tokens = text_tokens.to(device)
                    clip_embeddings = clip_model.encode_text(
                        text_tokens
                    ).float()
                    clip_embeddings_full = generate_clip_embeddings(
                        clip_model, text_tokens).float()
                    t5_embeddings_full = t5_model(cool_captions_text)
                    cool_captions_embeddings = torch.cat(
                        [clip_embeddings, torch.mean(t5_embeddings_full, dim=1)], 1)
                    cool_captions_embeddings_full = torch.cat([clip_embeddings_full,
                        t5_embeddings_full], 1)

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
                        sampled_text = sample(model, c=caption_embedding,
                            c_full=cool_captions_embeddings_full)  # [-1]
                        sampled_text = decode(vqmodel, sampled_text)
                        # sampled_text_ema = decode(vqmodel, sampled_text_ema)
                        for s in sampled_text:
                            cool_captions_sampled.append(s.cpu())
                            # cool_captions_sampled_ema.append(t.cpu())
                    print(
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
                torch.save(
                    model.state_dict(), f"models/{args.run_name}/model_{step}.pt"
                )
                torch.save(
                    optimizer.state_dict(),
                    f"models/{args.run_name}/model_{step}_optim.pt",
                )
            torch.save(model.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save(
                {"step": step, "losses": losses, "accuracies": accuracies},
                f"results/{args.run_name}/log.pt",
            )

        del images, image_indices, r, text_embeddings
        del noised_indices, mask, pred, loss, loss_adjusted, acc

        pbar.update(1)
        step += 1


def launch(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in args.devices])
    if len(args.devices) == 1:
        train(0, args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "33751"
        p = mp.spawn(train, nprocs=len(args.devices), args=(args,))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "run_name"
    args.model = "UNet"
    args.dataset_type = "webdataset"
    args.total_steps = 100_000
    args.batch_size = 4  # 22
    args.image_size = 512
    args.num_workers = 10
    args.log_period = 1000  # 5000
    args.extra_ckpt = 100_000
    args.ema = True
    args.ema_decay = 0.9999
    args.ema_update_steps = 100_000
    args.ema_model_path = 'ema_weights.ckpt'
    args.accum_grad = 1
    args.num_codebook_vectors = 8192
    args.log_captions = True
    args.finetune = False
    args.comparison_samples = 2 # 8
    args.cool_captions_text = [
        "a cat is sleeping",
        "a painting of a clown",
        # "a horse",
        # "a river bank at sunset",
        # "bon jovi playing a sold out show in egypt. you can see the great pyramids in the background",
        # "The citizens of Rome rebel against the patricians, believing them to be hoarding all of the food and leaving the rest of the city to starve",
        # "King Henry rouses his small, weak, and ill troops, telling them that the less men there are, the more honour they will all receive.",
        # "Upon its outward marges under the westward mountains Mordor was a dying land, but it was not yet dead. And here things still grew, harsh, twisted, bitter, struggling for life.",
    ]

    args.n_nodes = 1
    args.node_id = 0  # int(os.environ["SLURM_PROCID"])
    args.devices = [0]  # [0, 1, 2, 3, 4, 5, 6, 7]

    # Testing:
    # args.dataset_path = "gigant/oldbookillustrations_2"
    args.dataset_path = "laion/laion-coco"
    print("Launching with args: ", args)
    launch(args)
