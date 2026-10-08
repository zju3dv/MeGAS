# 3D Gaussian Ray Tracer

An OptiX-based differentiable 3D Gaussian Ray Tracer, inspired by the work [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/). See [gaussian-raytracing](https://github.com/fudan-zvg/gaussian-raytracing) for 3D reconstruction.

### Install
```bash
# clone the repo
git clone https://github.com/fudan-zvg/gtracer.git
cd gstracer

# use cmake to build the project for ptx file (for Optix)
rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../

# Install the package
pip install .
```

### Example usage
```bash
cd example
# Interactive viewer for 3DGS format point cloud
python renderer.py -p point_cloud.ply
```

### Acknowledgement

* Credits to [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [raytracing](https://github.com/NVlabs/instant-ngp).
* Credits to the original [3D Gaussian Ray Tracing](https://gaussiantracer.github.io/) paper.


## 📜 Citation
If you find this work useful for your research, please cite our github repo:
```bibtex
@misc{gu2024gtracer,
    title = {3D Gaussian Ray Tracer},
    author = {Gu, Chun and Zhang, Li},
    howpublished = {\url{https://github.com/fudan-zvg/gtracer}},
    year = {2024}
}
```
