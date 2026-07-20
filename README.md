# APL-VINS

ROS implementation of **APL-VIO**: Real-Time Visual-Inertial Odometry Using Adaptive Point-Line Features.

This repository corresponds to our paper accepted by *IEEE Transactions on Instrumentation and Measurement* (TIM).

## Related Paper

W. Li, L. Wu, X. Xiao, W. Xiao, and J. Bao, "APL-VIO: Real-Time Visual-Inertial Odometry Using Adaptive Point-Line Features," *IEEE Transactions on Instrumentation and Measurement*, 2026, doi: [10.1109/TIM.2026.3714602](https://doi.org/10.1109/TIM.2026.3714602).

If you use this code in your academic work, please cite the paper above.

### Prerequisites
- **System**
  - Ubuntu 20.04
  - ROS Noetic
- **Libraries**
  - [OpenCV 4.6.0 with opencv_contrib 4.6.0](https://blog.csdn.net/qq_44998513/article/details/133778446)
  - [Ceres Solver-1.14.0](http://ceres-solver.org/installation.html)

### Build
- **download the source package**
  - `mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src`
  - `git clone https://github.com/USST-LiLAB/APL-VINS.git`
- **build with OpenCV installed by yourself *(install in `/usr/local`)***
  - `gedit camera_model/CMakeLists.txt`
  - Modify `set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")`
  - `gedit aplvio/CMakeLists.txt`
  - Modify `set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")`
  - `gedit aplvio/include/visual/line_descriptor/CMakeLists.txt`
  - Modify `set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")`
  - *do NOT forget source your own cv_bridge workspace*
  - `source ~/cv_bridge/devel/setup.bash`
  - `cd ~/catkin_ws`
  - `catkin_make`

- **Notes**
  - ***The version of the OpenCV must be consistent with the version of OpenCV used by cv-bridge***

### Run
- **prepare output folder**
  - `mkdir -p ~/catkin_ws/src/APL-VINS/output`
- **launch**
  - `cd ~/catkin_ws`
  - `source devel/setup.bash`
  - `roslaunch aplvio euroc.launch dobag:=false`
- **play rosbag**
  - `rosbag play MH_01_easy.bag`

### Acknowledgements

This codebase is developed with reference to open-source VIO systems, especially:

- [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
- Related point-line VIO works such as PL-VINS

We thank the original authors for releasing their code.

### License

This project is released under the [GNU General Public License v3.0](LICENSE).

### Contact

- Lab / organization: [USST-LiLAB](https://github.com/USST-LiLAB)
- Paper contact: wangyan_li@usst.edu.cn
- Corresponding author: Wei Xiao (xiaow@sdju.edu.cn)

If you find this work useful, please consider starring the repository.
