# Paella
Conditional text-to-image generation has seen countless recent improvements in terms of quality, diversity and fidelity. Nevertheless, most state-of-the-art models require numerous inference steps to produce faithful generations, resulting in performance bottlenecks for end-user applications. In this paper we introduce Paella, a novel text-to-image model requiring less than 10 steps to sample high-fidelity images, using a speed-optimized architecture allowing to sample a single image in less than 500 ms, while having 573M parameters. The model operates on a compressed & quantized latent space, it is conditioned on CLIP embeddings and uses an improved sampling function over previous works. Aside from text-conditional image generation, our model is able to do latent space interpolation and image manipulations such as inpainting, outpainting, and structural editing.

![cover-figure](https://user-images.githubusercontent.com/117442814/201474789-a192f6ab-9626-4402-a3ec-81b8f3fd436c.png)

## Train your own Paella
The main file for training will be [paella.py](https://github.com/AmericanPresidentJimmyCarter/Paella). During training we use HF dataset.

### From Scratch
```
python3 paella.py
```

### License
The model code and weights are released under the [MIT license](https://github.com/AmericanPresidentJimmyCarter/Paella/blob/main/LICENSE).
