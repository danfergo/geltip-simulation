# GelTip Simulation

This repository contains the necessary code for executing the environmental setup and experiments as in our [GelTip Simulation paper](#). This includes the drivers for the GelTip sensor and the extended FDM 3D Printer, used for collecting the dataset, in real world and simulation, used in the experiments. The experiments include the final experiments for the dataset alignment and the *Sim2Real* classification task, and should be executed using Python 3. For more information about the work and links to download the used materials, visit [danfergo.github.io/geltip-simulation](https://danfergo.github.io/geltip-simulation/).

### Index of contents

| Component       | Description   |
| ------------- | ------------------|
| geltip                        | The GelTip driver component, and sim/real interfaces. OpenCV is used to access the geltip real webcam. |
| anet_a30                      | The FDM Printer driver component, and the sim and real interfaces. The real printer is controlled by issuing g-code commands through usb cable using pyserial.     |
| printer_extended              | The driver component for controlling the augmented printer, with two aditional Hi-TEC servos for the two additional degrees of freedom. The real servos are controlled through an arduino, running the driver arduino_driver.ino.  |
| data_collection_world         | The component were the environment is setup with the Extended Printer and GelTip | 
| data_collection_behaviour     | The control script to move the printer and grab and save frames from the geltip.  | 


## How to use
##### The experimental setup / world
Install [MuJoCo](https://mujoco.org/download), install [Yarok](https://pypi.org/project/yarok/) and clone the repository. Then,
```
python3 -m experimental_setup.data_collection.data_collection_world         # just view the world
python3 -m experimental_setup.data_collection.data_collection_behaviour     # run the world & collect data
```

##### The experiments
Install [pytorch](https://pytorch.org/) and clone the repository.
```
# todo 

```

