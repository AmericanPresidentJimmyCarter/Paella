import argparse
import sys

import torch

from utils import get_dataloader_laion_coco

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.batch_size = 2 # 22
args.total_steps = 12345
args.num_workers = 10
args.dataset_path = "laion/laion-coco"
dataset = get_dataloader_laion_coco(args)

batch_iterator = iter(dataset)
step = 0
epoch = 0
while step < args.total_steps:
    try:
        images, captions = next(batch_iterator)
    except StopIteration:
        epoch += 1
        print(f"Hit stop iteration, welcome to your next epoch: {epoch + 1}")
        batch_iterator = iter(dataset)
        images, captions = next(batch_iterator)
    except Exception as e:
        import traceback

        traceback.print_exc()
        continue
    
    print(len(images), captions)
    assert isinstance(images[0], torch.Tensor)
    assert isinstance(images[1], torch.Tensor)
    assert isinstance(captions[0], str)
    assert isinstance(captions[1], str)

    print('Test success')
    sys.exit()
