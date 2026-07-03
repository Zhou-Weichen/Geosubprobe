<div align="center">

# Understanding Geometric Representations in Self-Supervised Vision Transformers via Subspace Intervention

<p align="center">
🎉 <b>Accepted to ECCV 2026</b>
</p>

<p>
<a href="https://zhou-weichen.github.io/">Weichen Zhou</a><sup>1</sup> ·
<a href="https://zou-yawen.github.io/">Yawen Zou</a><sup>1</sup> ·
<a href="https://sites.google.com/view/gczjp/">Chunzhi Gu</a><sup>2</sup> ·
<a href="https://www.dr-lab.org/dong/">Ran Dong</a><sup>3</sup> ·
<a href="https://www.jaist.ac.jp/~xie/">Haoran Xie</a><sup>4</sup> ·
<a href="https://sites.google.com/view/chao-zhang/profile">Chao Zhang</a><sup>1†</sup>
</p>

<p>
<sup>1</sup> University of Toyama &nbsp;&nbsp;
<sup>2</sup> University of Fukui &nbsp;&nbsp;
<sup>3</sup> Chukyo University &nbsp;&nbsp;
<sup>4</sup> JAIST
</p>

<p>

[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-red)](https://arxiv.org/abs/2607.01987)
[![Project](https://img.shields.io/badge/Project-Homepage-blue)](https://github.com/Zhou-Weichen/Geosubprobe)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</p>

</div>

## Installation
```bash
git clone https://github.com/Zhou-Weichen/Geosubprobe.git
cd Geosubprobe

conda env create -f environment.yml
conda activate geosubprobe

pip3 install torch torchvision
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation
Please refer to [data_processing/README.md](data_processing/README.md) for detailed instructions on downloading and preprocessing all datasets.






## Acknowledgements
This project is largely built upon [Probe3D](https://github.com/mbanani/probe3d). We thank the authors for open-sourcing their excellent implementation, which served as the foundation for our work.
