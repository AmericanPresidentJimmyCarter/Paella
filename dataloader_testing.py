import argparse
import sys
import requests

import torch

resp_dict = None
try:
    resp = requests.post(url='http://127.0.0.1:4455/conditionings', json={
        'captions': ['foo', 'bar'],
    })
    resp_dict = resp.json()
except Exception:
    import traceback
    traceback.print_exc()
assert resp_dict is not None
assert 'flat' in resp_dict
assert 'full' in resp_dict
assert 'flat_uncond' in resp_dict
print(len(resp_dict['flat_uncond']))
assert 'full_uncond' in resp_dict
print(len(resp_dict['full_uncond']))

for i in range(4):
    resp_dict = None
    try:
        resp = requests.post(url='http://127.0.0.1:4456/batch')
        resp_dict = resp.json()
    except Exception:
        import traceback
        traceback.print_exc()
    assert resp_dict is not None
    assert 'captions' in resp_dict
    assert 'images' in resp_dict
    assert 'conditioning_flat' in resp_dict
    assert 'conditioning_full' in resp_dict
    assert 'unconditioning_flat' in resp_dict
    assert 'unconditioning_full' in resp_dict

print(len(resp_dict['conditioning_flat']))
print(len(resp_dict['conditioning_full']))

print(len(resp_dict['unconditioning_flat']))
print(len(resp_dict['unconditioning_full']))
print('Server appears to be working.')
