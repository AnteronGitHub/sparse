# Sparse

This repository contains source code for Stream Processing Architecture for Resource Subtle Environments (or just
Sparse for short). Additionally, sample applications utilizing Sparse for deep learning can be found in examples
directory.

```
pip install sparse-framework
```

## Example Applications

The repository includes example applications (in the [examples directory](https://github.com/AnteronGitHub/sparse/blob/master/examples)). The applications are tested
tested with the following devices and the following software:

| Device            | JetPack version | Python version | PyTorch version | Docker version | Base image                                     | Docker tag suffix |
| ----------------- | --------------- | -------------- | --------------- | -------------- | ---------------------------------------------- | ------------------ |
| Jetson AGX Xavier | 5.0 preview     | 3.8.10         | 1.12.0a0        | 20.10.12       | nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3 | jp50               |
| Lenovo ThinkPad   | -               | 3.8.12         | 1.11.0          | 20.10.15       | pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime | amd64              |

See [how to deploy the example applications with Kubernetes](https://github.com/AnteronGitHub/sparse/blob/master/k8s).

# Citation in articles

The following article introduces this design (corresponds to version `v1.0.0-rc2`):
```
@INPROCEEDINGS{10403079,
  author={Vainio, Antero and Mudvari, Akrit and Kiedanski, Diego and Tarkoma, Sasu and Tassiulas, Leandros},
  booktitle={2023 IEEE 7th International Conference on Fog and Edge Computing (ICFEC)},
  title={Fog Computing for Deep Learning with Pipelines},
  year={2023},
  volume={},
  number={},
  pages={64-72},
  keywords={Training;Program processors;Computational modeling;Pipelines;Hardware;Task analysis;Edge computing;fog computing;stream processing systems;deep learning},
  doi={10.1109/ICFEC57925.2023.00017}}
```

