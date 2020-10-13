# SegNet Implementation by Pytorch #
ehis repo is a Pytorch-lightning re-implementation of [SegNet](https://arxiv.org/abs/1511.00561). The code was forked from []().

## To run:
* Load Pascal VOC2012 dataset. 
```bash
cd pytorch-segnet/
bash load_voc2012.sh
```
* Train and validate the model. This will download pretrained VGG16 weights into the encoder section of SegNet.
```bash
python main.py
```
* To download and evalute with my pretrained weights (TODO) 
```
```

## Requirements:
```
torch
pytorch-lightning
numpy
matplotlib
```

## License
```
MIT
```
