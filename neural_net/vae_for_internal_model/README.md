
# Environment Setup
### Ubuntu 20.04, ROS Noetic

---

### Create Conda Environment 

Create a conda environment using an environment file that is prepared at `config/conda`.
```
$ conda env create --file config/conda/environment.yaml
$ conda env create --file config/conda/keras.yaml
$ conda env create --file config/conda/torch.yaml
```
---
### Activate Conda Environment for Train DNN
Activate the `keras` environment for Keras.

```
$ conda activate keras
```
Activate the `torch` environment for Pytorch.
The environment that should be used when running `*_pytorch.py` in `neural_net/vae_for_internal_model`.
```
$ conda activate torch
```
---
### Build
For some reason, building catkin_make throws an error.
However, if I type the same command one more time, it builds fine. 
```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

---
When writing Python code, you can do this by writing the following at the top of your code
```python
#!/usr/bin/env python3
```
---
## Dataset for Internal Model
The dataset structure is as follows. 
The data used to train the internal model are `'image_fname'`, `'tar_image_fname'`, `'tar_steering_angle'`, `'tar_vel'`, and `'tar_time'`.

```python
csv_header = [
        'image_fname', 'steering_angle', 'throttle', 'brake', 'linux_time', 
        'vel', 'vel_x', 'vel_y', 'vel_z', 'pos_x', 'pos_y', 'pos_z', 
        'tar_image_fname', 'tar_steering_angle', 'tar_vel', 'tar_time'
    ]
```
`'tar_image_fname'` is the name of the image collected after `'tar_time'` seconds from the time of collecting `'image_fname'`.
`'tar_steering_angle'` and `'tar_vel'` are the average values of steering angle and velocity from the time of collecting `'image_fname'` to the time of collecting `'tar_image_fname'`.

---