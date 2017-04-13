# Super resolution using subpixel CNNs implementation with Tensorflow

To train
```bash
python main.py --dataset celebA --is_train True --is_crop True
```

## Stuff to still do
1. Try it on simple VAE architecture
2. Make it work for multi-size multi-aspect ratio images

## Acknowledgements
1. [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
2. [Subpixel implementation](https://github.com/Tetrachrome/subpixel)

## References

1.  [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158). By Shi et. al.  
2. [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901). By Zeiler and Fergus.  
3. [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285). By Dumoulin and Visin.

## Further reading
Alex J. Champandard made a really interesting analysis of this topic in this [thread](https://twitter.com/alexjc/status/782499923753304064).   
For discussions about differences between phase shift and straight up `resize` please see the companion [notebook](https://github.com/Tetrachrome/subpixel/blob/master/ponynet.ipynb) and this [thread](https://twitter.com/soumithchintala/status/782603117300965378).
