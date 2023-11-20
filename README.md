# CoMoDO

## Control Motion Design Optimization 

Suite of parametrized controller and simulator for codesign of robots.


---

<p align="center">
  <b>⚠️ REPOSITORY UNDER DEVELOPMENT ⚠️</b>
  <br>We cannot guarantee stable API
</p>

---

## Installation 
This library depends on 

- [``casadi``](https://web.casadi.org/)
- [``numpy``](https://numpy.org/)
- [``idyntree``](https://github.com/robotology/idyntree)
- [``bipedal-locomotion-framework``](https://github.com/ami-iit/bipedal-locomotion-framework)
- [``adam-robotics``](https://github.com/ami-iit/ADAM/tree/main)
- [``mujoco``](https://mujoco.org/)
- [``mujoco-python-viewer``](https://github.com/rohanpsingh/mujoco-python-viewer)
- [``matplotlib``](https://matplotlib.org/stable/)
- [``gitpython``](https://github.com/gitpython-developers/GitPython)
- [``urchin``](https://github.com/fishbotics/urchin)

To install you can use the following commands


```
conda create -n comododev -c conda-forge adam-robotics idyntree bipedal-locomotion-framework mujoco mujoco-python numpy mujoco-python-viewer matplotlib gitpython urchin
conda activate comododev
pip install --no-deps git+https://github.com/CarlottaSartore/urdf-modifiers.git@scalar_modification 
pip install --no-deps -e .

```

## Usage 

Take a look at the [examples](https://github.com/ami-iit/comodo/tree/main/examples) folder! 

### Maintainer

This repository is maintained by 
|                                                              |                                                      |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| [<img src="https://user-images.githubusercontent.com/56030908/135461492-6d9a1174-19bd-46b3-bee6-c4dbaea9e210.jpeg" width="40">](https://github.com/S-Dafarra) | [@CarlottaSartore](https://github.com/CarlottaSartore) |



