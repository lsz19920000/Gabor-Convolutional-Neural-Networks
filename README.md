# Gabor-Convolutional-Neural-Networks
 
#A demo for GCN on MNIST dataset using torch7
To run this demo, you should  install these dependencies:

`luarocks install torchnet`

`luarocks install optnet`

#install GCN:
```bash
cd $DIR/GCN
bash install.sh
```

#run this demo:
```bash
cd $DIR/MNIST_demo
bash ./scripts/Train_MNIST.sh
```

#Acknowledgement
```
This demo is partially referenced to the code of Orientation Response Networks(ORN,`http://zhouyanzhao.github.io/ORN/`)
If you use this demo please cite our paper and ORN. 

bibtex:

@article{GCN,
  title={Gabor Convolutional Networks},
  year={2017},
}


@INPROCEEDINGS{Zhou2017ORN,
  title={Oriented Response Networks},
  author={Zhou, Yanzhao and Ye, Qixiang and Qiu, Qiang and Jiao, Jianbin},
  booktitle = {CVPR}
  year={2017},
}
```

