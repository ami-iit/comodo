# CoMoDO

**Control Motion Design Optimization**

**CoMoDO** is a suite of parametrized controllers and simulators for codesign of robots.


---

<p align="center">
  <b>⚠️ REPOSITORY UNDER DEVELOPMENT ⚠️</b>
  <br>We cannot guarantee stable API
</p>

---

## Table of contents

- [⚙️ Dependencies](#-dependencies)
- [💾 Installation](#-installation)
  - [🐍 Installation with conda](#-installation-with-conda)
  - [📦 Installation with pixi](#-installation-with-pixi)
- [🚀 Usage](#-usage)
- [:construction_worker: Maintainer](#construction_worker-maintainer)


## ⚙️ Dependencies

This library depends on

- [``casadi``](https://web.casadi.org/)
- [``numpy``](https://numpy.org/)
- [``idyntree``](https://github.com/robotology/idyntree)
- [``bipedal-locomotion-framework``](https://github.com/ami-iit/bipedal-locomotion-framework)
- [``adam-robotics``](https://github.com/ami-iit/ADAM)
- [``matplotlib``](https://matplotlib.org/stable/)
- [``urllib3``](https://urllib3.readthedocs.io/en/stable/)
- [``urchin``](https://github.com/fishbotics/urchin)

And, optionally, on:

- [``mujoco``](https://mujoco.org/) and [``mujoco-python-viewer``](https://github.com/rohanpsingh/mujoco-python-viewer)
- [``jaxsim``](https://github.com/ami-iit/jaxsim)
- [`drake`](https://drake.mit.edu/)
- [`hippopt`](https://github.com/ami-iit/hippopt.git)

## 💾 Installation

### 🐍 Installation with conda

To install comodo in a conda environment, you can use the following commands

```bash
conda create -n comododev -c conda-forge adam-robotics bipedal-locomotion-framework=0.19.0 mujoco-python-viewer matplotlib urllib3 urchin notebook jaxsim

conda activate comododev
pip install --no-deps git+https://github.com/CarlottaSartore/urdf-modifiers.git@scalar_modification
pip install --no-deps -e .

```

#### With hippopt

To work in comodo with [`hippopt`](https://github.com/ami-iit/hippopt.git) requires to install also the following packages:

```bash
conda install -c conda-forge -c robotology casadi pytest liecasadi  meshcat-python ffmpeg-python
pip install --no-deps git+https://github.com/ami-iit/hippopt.git
```

#### With Drake

To use [`drake`](https://drake.mit.edu/) as the simulator backend requires the following additional dependencies:

```bash
conda install meshio tqdm
pip install drake git+https://github.com/ami-iit/amo_urdf
```

### 📦 Installation with pixi

An alternative and easy way to use comodo is with [`pixi`](https://pixi.sh/latest/) package manager. It automatically handles the creation and activation of virtual environments in which to use the different simulators that comodo supports.

At the moment there is an environment associated with each simulator backend, namely: 
-  `mujoco`: for mujoco simulator
- `jaxsim`: for jaxsim simulator
- `drake`: for drake simulator 
- `all` for all the simulators

To activate one of these environments in a terminal run:

```bash
pixi shell -e <environment-name>
```

It is also possible to run directly a command in one of these environments using:

```bash
pixi run -e <environment-name> python <script-filename.py>
```

For example, it is possible to run the Jupyter notebooks in the [examples](./examples) folder by just executing:

```bash
pixi run -e all jupyter notebook
```

and then running the examples you prefer in Jupyter.

## 🚀 Usage

Take a look at the [examples](./examples) folder!

## :construction_worker: Maintainer

This repository is maintained by
|                                                              |                                                      |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| <img src="https://user-images.githubusercontent.com/56030908/135461492-6d9a1174-19bd-46b3-bee6-c4dbaea9e210.jpeg" width="40"> | [@CarlottaSartore](https://github.com/CarlottaSartore) |
