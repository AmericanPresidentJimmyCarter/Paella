import base64
import io
from typing import Dict, List, Union

from fastapi import FastAPI, HTTPException

import argparse
import sys

import torch
from pydantic import BaseModel


from data import tensor_to_b64_string, get_dataloader_laion_coco


CONDITIONING_DEVICE = 'cuda:0'


class ConditioningRequest(BaseModel):
    captions: List[str]


class BatchResponse(BaseModel):
    images: str
    captions: List[str]
    conditioning_flat: str
    conditioning_full: str
    unconditioning_flat: str
    unconditioning_full: str


class ConditioningResponse(BaseModel):
    flat: str
    full: str


class Arguments:
    batch_size = 12
    num_workers = 16
    dataset_path = "laion/laion-coco"
    # cache_dir = "/home/user/.cache"  # cache_dir for models


dataset = get_dataloader_laion_coco(Arguments())


batch_iterator = iter(dataset)
epoch = 0

app = FastAPI()

@app.post("/batch")
def batch() -> BatchResponse:
    global batch_iterator
    global epoch

    try:
        images, captions = next(batch_iterator)
        flat = captions.get('flat')
        full = captions.get('full')
        flat_uncond = captions.get('flat_uncond')
        full_uncond = captions.get('full_uncond')
        captions = captions.get('captions')
        resp = BatchResponse(
            captions=captions,
            images=tensor_to_b64_string(images),
            conditioning_flat=tensor_to_b64_string(flat),
            conditioning_full=tensor_to_b64_string(full),
            unconditioning_flat=tensor_to_b64_string(flat_uncond),
            unconditioning_full=tensor_to_b64_string(full_uncond),
        )
        return resp
    except StopIteration:
        epoch += 1
        print(f"Hit stop iteration, welcome to your next epoch: {epoch + 1}")
        batch_iterator = iter(dataset)
        images, captions = next(batch_iterator)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))