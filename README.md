# Paellaaa

A conditional text-to-image generative model based on [Paella](https://github.com/dome272/Paella) by Jimmy and Hlky. It incorporates cross attention in addition to using LayerNorm conditioning. It uses both the conditionings from CLIP and T5, similar to the dual conditioning from [Kadinsky-2.0](https://github.com/ai-forever/Kandinsky-2.0).

```
  ┌───────────────────────────────────────┐
  │                                       │
  │  ┌──────────┐           ┌─────────┐   │
  │  │          ├──────────►│         │   │
  │  │          │           │ Layer   ├───┘
  │  │ OpenCLIP │           │ Norm    │
  │  │          │        ┌─►│         │
  │  │          ├─────┐  │  │         ├────────┐
  │  └──────────┘     │  │  └─────────┘        │
  │                   │  │                     │
  │  ┌──────────┐     │  │  ┌─────────┐        │
  │  │          ├─────┼──┘  │         │        │
  │  │          │     │     │ Cross   │        │
  │  │ T5       │     └────►│ Attn    │        │
  │  │          │           │         │        │
  │  │          ├──────────►│         │        │
  │  └──────────┘           └─┬─────┬─┘        │
  │                           │     │          │
  │                  ┌────────┘     └──────┐   │
  │   UNet Down      │      UNet Up        │   │
  │  ┌────────────┐  │     ┌────────────┐  │   │
  ├─►│            ├──┼─────►            │◄─┼───┤
  │  └─────▲──────┘  │     └─────┬──────┘  │   │
  │        │         │           │         │   │
  │    ┌───┴────┐    │       ┌───▼────┐    │   │
  ├───►│        │    │       │        │◄───┼───┤
  │    └───▲────┘    │       └───┬────┘    │   │
  │        │         │           │         │   │
  │      ┌─┴──┐      │         ┌─▼──┐◄─────┘   │
  └─────►│    │◄─────┘         │    │          │
         └─▲──┘                └─┬──┘◄─────────┘
           │                     │
      ┌────┴─────┐          ┌────▼─────┐
      │          │          │          │
      │Latent in │          │Latent Out│
      │          │          │          │
      │          │          │          │
      │          │          │          │
      └──────────┘          └──────────┘
```

Weights are forthcoming.

## Train your own Paella
The main file for training will be [paella.py](https://github.com/AmericanPresidentJimmyCarter/Paellaaa). During training we use HF dataset.

### From Scratch
```
python3 paella.py
```

### License
The model code and weights are released under the [MIT license](https://github.com/AmericanPresidentJimmyCarter/Paella/blob/main/LICENSE).
