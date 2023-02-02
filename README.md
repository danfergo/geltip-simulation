# GelTip Simulation

This repository contains the necessary code for executing the environmental setup and experiments as in our [GelTip Simulation paper](#). This includes the interfaces for the GelTip sensor and the extended FDM 3D Printer, used for collecting the dataset used in the experiments (collected in the real world and simulation). For more information about the work and links to download the used materials, visit [danfergo.github.io/geltip-simulation](https://danfergo.github.io/geltip-simulation/).

### Index
| Packages       | Description   |
| ------------- | ------------------|
| dfgiatk                       | Collection of functions to run experiments, train neural nets, etc |
| experimental_setup            | Yarok components to control the 3D Printer and GelTip sensor (real and sim). Scripts to build light fields etc for the Geltip sim model.    |
| geltip_dataset                | Dataset  |
| data_collection_world         | The component were the environment is setup with the Extended Printer and GelTip | 
| data_collection_behaviour     | The control script to move the printer and grab and save frames from the geltip.  | 


#### Yarok components
| Component       | Description   |
| ------------- | ------------------|
| GelTip                        | The GelTip driver component, and sim/real interfaces. OpenCV is used to access the geltip real webcam. |
| AnetA30                      | The FDM Printer driver component, and the sim and real interfaces. The real printer is controlled by issuing g-code commands through usb cable using pyserial.     |
| PrinterExtended              | The driver component for controlling the augmented printer, with two aditional Hi-TEC servos for the two additional degrees of freedom. The real servos are controlled through an arduino, running the driver arduino_driver.ino.  |
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

