import serial
from yarok import ConfigBlock, component, interface
from yarok.platforms.mjc import InterfaceMJC

from experimental_setup.anet_a30.anet_a30 import AnetA30
from math import pi

import time


def deg2rad(deg):
    return deg / (180 / pi)


def sae(q1, q2):
    return sum([abs(q1[i] - q2[i]) for i in range(len(q1))])


@interface(
    defaults={
        'angular_velocity': 1
    }
)
class PrinterExtendedInterfaceMJC:
    def __init__(self, interface: InterfaceMJC, config: ConfigBlock):
        self.interface = interface
        self.gear = 50
        self.speed = config['angular_velocity'] * self.gear * (2 * pi)
        self.q = [0, 0]
        self.target_q = self.q
        self.start_t = time.time()
        self.delta_t = 0
        self.start_q = None
        self.last_q = None
        self.stopped_steps = 0

    # moves the printer to the initial position
    # implemented to match the same api as the real printer.
    # it isn't necessary
    def home_servos(self):
        pass

    def move_servos(self, angles):
        self.start_q = self.target_q

        # converts from deg into rad, and send control to the servos
        self.target_q = [deg2rad(angles[i]) * self.gear for i in range(len(angles))]
        self.start_t = time.time()
        self.delta_t = max([abs(self.target_q[i] - self.start_q[i]) / self.speed for i in range(2)])

    def servos_at(self, angles):
        # reads sensor angles, converts from deg into rad,
        # and returns true if the SAD < 0.0006
        sensor_data = self.interface.sensordata()

        return self.stopped_steps > 10 and \
               sum([abs((deg2rad(angles[i]) * self.gear) - sensor_data[i]) for i in range(2)]) < 1e-6

    def step(self):
        self.last_q = self.q
        self.q = self.interface.sensordata()
        now = time.time()

        if self.start_q is not None:
            elapsed = (now - self.start_t)
            progress = min(1, elapsed / self.delta_t) if self.delta_t > 0 else 1
            target = [(1 - progress) * self.start_q[i] + progress * self.target_q[i] for i in range(2)]
        else:
            target = self.target_q

        [self.interface.set_ctrl(a, target[a]) for a in range(len(self.interface.actuators))]

        if sae(self.last_q, self.q) < 1e-6:
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0


@interface(
    defaults={
        'serial_path': '/dev/ttyUSB1',
        'serial_port': 9600
    }
)
class PrinterExtendedInterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.ser_con = serial.Serial(config['serial_path'], config['serial_port'], timeout=1)

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
                <material name="black_plastic" rgba=".1 .1 .1 1"/>
                <material name="black_metal" rgba=".3 .3 .3 1"/>

                <!-- mounts -->
                <mesh name="object_mount" file="meshes/mount_object.stl" scale="0.001 0.001 0.001"/>
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
                            <!-- x: y: z: --> 
                            <body name="a2_body" pos="-0.032 -0.04 -0.016" xyaxes="1 0 0 0 -1 0">
                                <body>
                                    <joint name="a1"
                                           armature="1"
                                           damping="100"
                                           type="hinge"
                                           axis="0 -1 0"/>
                                    
                                    <geom type="mesh"
                                          pos="0 0 0"
                                          mesh="object_mount"
                                          material="black_plastic"/>
                                    
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
                                  pos="0.17 0.165 0.004"
                                  rgba=".8 .68 0.5 1"/>
                                  
                            <body pos="0.153 0.128 0.046" xyaxes="0 1 0 0 0 1">
                                <!-- let's pretend servo -->
                                <geom type="box"
                                  zaxis='0 1 0'
                                  pos="0.01 0.0 -0.065"
                                  size=".025 .018 .0125"
                                  material="black_plastic"/>
                                  
                              <!-- let's pretend servo mount -->
                              <geom type="box"
                                  pos="0.01 -.014 -0.065"
                                  size=".04 .001 .0125"
                                  material="black_metal"/>
                              <geom type="box"
                                  pos="0.01 -.038 -0.065"
                                  size=".04 .001 .0125"
                                  material="black_metal"/>
                              <geom type="box"
                                  zaxis='0 1 0'
                                  pos="0.01 -.026 -0.053"
                                  size=".04 .001 .0125"
                                  material="black_metal"/>
                                  
                                
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
                                           armature="1"
                                           damping="100"
                                           axis="0 0 -1"/>
                                           
                                       <!-- lets the sim env work without sensor -->
                                       <geom type="cylinder" 
                                                density="0.1" 
                                                pos="0 0 -0.045 "
                                                size="0.01 0.001"/> 
                                  </body>
                            </body>
                    </body>
                </anet_a30>
            </worldbody>
            <actuator>
                  <position name="a1" gear="50" joint="a1" forcelimited="true" forcerange="-10 10" kp='10'/>
                  <position name="a2" gear="50" joint="a2" forcelimited="true" forcerange="-10 10" kp='10'/>
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
        return lambda: self.is_at(position, angles)
