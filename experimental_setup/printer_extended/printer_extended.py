import serial
from yarok import ConfigBlock, component, interface
from yarok.platforms.mjc import InterfaceMJC

from experimental_setup.anet_a30.anet_a30 import AnetA30
from math import pi


def deg2rad(deg):
    return deg / (180 / pi)


@interface()
class PrinterExtendedInterfaceMJC:
    def __init__(self, interface: InterfaceMJC):
        self.interface = interface
        self.gear = 100

    # moves the printer to the initial position
    # implemented to match the same api as the real printer.
    # it isn't necessary
    def home_servos(self):
        pass

    def move_servos(self, angles):
        # converts from deg into rad, and send control to the servos
        [self.interface.set_ctrl(i, deg2rad(angles[i]) * self.gear) for i in range(len(angles))]

    def servos_at(self, angles):
        # reads sensor angles, converts from deg into rad,
        # and returns true if the SAD < 0.0006
        sensor_data = self.interface.sensordata()
        return sum([abs((deg2rad(angles[i]) * self.gear) - sensor_data[i]) for i in range(2)]) < 0.0006


@interface(

)
class PrinterExtendedInterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.ser_con = serial.Serial('/dev/ttyUSB1', 9600, timeout=1)

        self.initialized = False
        self.executing_cmd = False
        self.last_exec_cmd = ''
        self.angles = None

    def home_servos(self):
        self.move_servos((0, 0))

    def move_servos(self, angles):
        cmd = str(angles[0]) + ";" + str(angles[1]) + '; '
        self.ser_con.write(str.encode(cmd))
        self.ser_con.flush()

        self.last_exec_cmd = 'move'
        self.executing_cmd = True
        self.angles = angles

    def is_moving(self):
        return self.executing_cmd and self.last_exec_cmd == 'move'

    def servos_at(self, angles):
        if not self.is_moving() and self.angles == angles:
            return True
        return False

    def is_ready(self):
        if not self.initialized:
            return False
        elif not self.executing_cmd:
            if self.last_exec_cmd == '':
                self.home_servos()
            else:
                return True
        return False

    def step(self):
        ln = self.ser_con.readline().decode().rstrip()

        if not self.initialized:
            if ln == 'Hi!':
                self.initialized = True
        else:
            if self.executing_cmd:
                if ln == 'ok.':
                    self.executing_cmd = False


@component(
    tag="printer_extended",
    defaults={
        'interface_mjc': PrinterExtendedInterfaceMJC,
        'interface_hw': PrinterExtendedInterfaceHW,
    },
    components=[
        AnetA30
    ],
    # language="xml"
    template="""
        <mujoco>
            <asset>
                <material name="black_plastic" rgba=".3 .3 .3 1"/>
                
                <!-- mounts -->
                <mesh name="tiny_object_set_mount" file="meshes/mount_object.stl" scale="0.001 0.001 0.001"/>
                <mesh name="geltip_printer_mount" file="meshes/geltip_printer_mount.stl" scale="0.001 0.001 0.001"/>
            </asset>
            <worldbody>
                <anet_a30 name="printer">
                    <body parent="printer_head">
                             <geom type="box"
                                  pos="-0.01 -0.014 -0.015"
                                  size=".025 .018 .0125"
                                  material="black_plastic"/>
    
                            <!-- Actuator 2: indenter orientation -->
                            <body name="a2_body" 
                            pos="-0.025 -0.04 -0.015" 
                            xyaxes="1 0 0 0 -1 0">
    
                                <body>
                                    <joint name="a1"
                                           damping="10"
                                           frictionloss="10"
                                           type="hinge"
                                           axis="0 -1 0"/>
    
                                    <!-- indenter -->
                                    <body pos="0 0.0108 0" name="indenter_mount">
                                        
                                    </body>                                      
                                </body>
                            </body>
                        </body>
                
                    <body parent="printer_bed">
                            <!-- wooden board (base) -->
                            <geom type="box"
                                  size="0.1085 0.165 0.003"
                                  pos="0.19 0.165 0.004"
                                   rgba=".8 .68 0.5 1"/>
                                  
                            <body pos="0.153 0.128 0.046" xyaxes="0 1 0 0 0 1">
                                <!-- plastic geltip mount -->
                                <geom type="mesh"
                                  pos="0 0 0.003"
                                  mesh="geltip_printer_mount"
                                  material="black_plastic"
                                  friction="0.4 0.4 0.8"/>
                                   
                                  <!-- the mount that rotates the sensor --> 
                                  <body name="geltip_mount">
                                      <joint name="a2"
                                           type="hinge"
                                           frictionloss="10"
                                           damping="10"
                                           axis="0 0 1"/>
                                           
                                       <!-- lets the sim env work without sensor -->
                                       <geom type="cylinder" 
                                                density="0.1" 
                                                pos="0 0 -0.05 "
                                                size="0.01 0.001"/> 
                                                
                                  </body>
                            </body>
                    </body>
                </anet_a30>
            </worldbody>
            <actuator>
                  <position name="a1" gear="100" joint="a1" forcelimited="true" forcerange="-1.05 1.05" kp='1000'/>
                  <position name="a2" gear="100" joint="a2" forcelimited="true" forcerange="-1 1" kp='1000'/>
            </actuator>
            <sensor>
                   <actuatorpos name="a1" actuator="a1"/>
                   <actuatorpos name="a2" actuator="a2"/>
            </sensor>   
        </mujoco>
    """,

)
class PrinterExtended:

    def __init__(self, printer: AnetA30):
        self.printer = printer

    def home_servos(self):
        # implemented by the interface
        pass

    def move_servos(self, angles):
        # implemented by the interface
        pass

    def servos_at(self, angles):
        # implemented by the interface
        pass

    def is_at(self, position=None, angles=None):
        if angles is not None and not self.servos_at(angles):
            return False

        return (position is None) or self.printer.is_at(position)

    def move(self, position=None, angles=None):
        if angles is not None:
            self.move_servos(angles)

        if position is not None:
            self.printer.move(position)
