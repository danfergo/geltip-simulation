from yarok import ConfigBlock, component, interface
from yarok.platforms.mjc import InterfaceMJC

import serial


@interface()
class AnetA30InterfaceMJC:

    def __init__(self, interface: InterfaceMJC):
        self.interface = interface
        self.gear = 100
        self.MAX_X = 320
        self.MAX_Y = 320
        self.MAX_Z = 420

        # includes gear ratio and millimeters to meters conversion.
        self.mm2ctrl = self.gear / 1000.0

    # moves the printer to the initial position
    # implemented to match the same api as the real printer.
    # it isn't necessary for simulation
    def home(self):
        pass

    def move(self, p):
        x, y, z = [
            max(0, min(p[0], self.MAX_X)),
            max(0, min(p[1], self.MAX_Y)),
            max(0, min(p[2], self.MAX_Z))
        ]

        self.interface.set_ctrl('ax', x * self.mm2ctrl)
        self.interface.set_ctrl('ay', y * self.mm2ctrl)
        self.interface.set_ctrl('az', z * self.mm2ctrl)

    def is_at(self, position):
        sensor_data = [d / self.mm2ctrl for d in self.interface.sensordata()]
        # print(sum([abs(position[i] - sensor_data[i]) for i in range(3)]))
        return sum([abs(position[i] - sensor_data[i]) for i in range(3)]) < 0.25  # 0.25 mm

    def step(self):
        pass


@interface(
    defaults={
        'serial_path': '/dev/ttyUSB0'
    }
)
class AnetA30InterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.ser_con = serial.Serial(config['serial_path'], 115200, timeout=1)

        self.MAX_X = 320
        self.MAX_Y = 320
        self.MAX_Z = 420
        self.initialized = False
        self.position = None

        self.uninitialized_empty_counter = 0  # used to keep track of the initialization
        self.wait_until_initialize = 5
        self.initialized = False

        self.executing_cmd = False
        self.last_exec_cmd = ''

    def gcode_parse_pos(self, res):
        arr = res.split(' ')
        pos_d = {a.split(':')[0]: float(a.split(':')[1]) for a in arr}
        return pos_d['X'], pos_d['Y'], pos_d['Z']

    def send_cmd(self, cmd, args=None):
        if not self.initialized:
            return False
        full_cmd = cmd + " " + args if args is not None else cmd
        self.ser_con.write(str.encode(full_cmd + ' \n'))
        self.ser_con.flush()

        self.executing_cmd = True
        self.last_exec_cmd = cmd

        return True

    def home(self):
        # implemented by the interface
        pass

    def is_at(self, position):
        if not self.is_moving() and \
                (position is not None) and (self.position is not None) and \
                sum([abs(position[i] - self.position[i]) for i in range(3)]) < 0.25:  # 0.25 mm
            return True
        return False

    def is_moving(self):
        return (self.last_exec_cmd == 'G0' or self.executing_cmd == 'G28') \
               and self.executing_cmd

    def move(self, position: [float, float, float]):
        x, y, z = round(position[0], 2), round(position[1], 2), round(position[2], 2)
        x = max(0, min(x, self.MAX_X))
        y = max(0, min(y, self.MAX_Y))
        z = max(0, min(z, self.MAX_Z))
        if self.send_cmd("G0", "X%.2f Y%.2f Z%.2f" % (x, y, z)):
            self.position = None

    def is_ready(self):
        if not self.initialized:
            return False
        elif not self.executing_cmd:
            if self.last_exec_cmd != 'G28':
                # home the printer
                self.send_cmd('G28')
            else:
                return True
        return False

    def step(self):
        ln = self.ser_con.readline().decode().rstrip()
        if not self.initialized:
            self.uninitialized_empty_counter = self.uninitialized_empty_counter + 1 if ln == '' else 0
            if self.uninitialized_empty_counter >= self.wait_until_initialize:
                self.initialized = True
        else:
            if self.executing_cmd:
                if ln == 'ok':
                    self.executing_cmd = False
                elif self.last_exec_cmd == 'M114' and ln != '':
                    self.position = self.gcode_parse_pos(ln)
            else:
                if self.last_exec_cmd != 'M114':
                    self.send_cmd('M114')


@component(
    tag="anet_a30",
    defaults={
        'interface_mjc': AnetA30InterfaceMJC,
        'interface_hw': AnetA30InterfaceHW
    },
    # language=xml
    template="""
    <mujoco>
    <asset>
        <mesh name="fdm_printer_bottom_frame" file="meshes/bottom_frame.stl" scale="0.0011 0.0011 0.0011"/>
        <mesh name="fdm_printer_side1_frame" file="meshes/side1_frame.stl" scale="0.0011 0.0011 0.0011"/>
        <mesh name="fdm_printer_side2_frame" file="meshes/side2_frame.stl" scale="0.0011 0.0011 0.0011"/>


        <mesh name="fdm_printer_bed" file="meshes/bed2.stl" scale="0.0011 0.0011 0.0011"/>
        <mesh name="fdm_printer_x_axis" file="meshes/x_axis.stl" scale="0.0011 0.0011 0.0011"/>

        <material name="black_metal" rgba=".2 .2 .2 1" specular="0.95"/>

    </asset>
    <worldbody>
        <body>

            <!-- External frame -->
            <body pos="0 0 0.09">
                <geom type="mesh"
                      mesh="fdm_printer_bottom_frame"
                      material="black_metal"/>
                <geom type="mesh"
                      mesh="fdm_printer_side1_frame"
                      material="black_metal"/>
                <geom type="mesh"
                      mesh="fdm_printer_side2_frame"
                      material="black_metal"/>
                <geom type="mesh"
                      mesh="fdm_printer_x_axis"
                      material="black_metal"/>
            </body>

            <!-- Z-AXIS: horizontal metal bar -->
            <body name="zaxis_body" pos="0 -0.025 0.0634">

                <joint name="zaxis"
                       type="slide"
                       armature="1000"
                       frictionloss="1000"
                       axis="0 0 1"/>

                <geom type="mesh"
                      mesh="fdm_printer_x_axis"
                      material="black_metal"
                      fitscale="0.001"
                      pos="0 0 -0.498"/>

                <!-- x-axis -->
                <body name="xaxis_body">
                    <joint name="xaxis"
                           type="slide"
                           armature="1000"
                           frictionloss="1000"
                           axis="1 0 0"/>

                    <!-- printer head-->
                    <body pos="-0.002 0 0.05" name="printer_head">
                        <geom type="box"
                              pos="0.001 0.01 0.005"
                              size=".04 .002 .035"
                              material="black_metal"/>
                        <!-- <geom type="box"
                              pos="-0.01 -0.014 -0.015"
                              size=".025 .018 .0125"
                              material="black_plastic"/> -->
                              
                              
                    </body>
                </body>
            </body>

            <body name="printer_bed" pos="0 0 0.08">
                <joint name="yaxis"
                       type="slide"
                       armature="1000"
                       frictionloss="1000"
                       axis="0 -1 0"/>

                <!-- area/size of the real printer bed -->
                <!-- <geom type="box"
                      name="fake_bed"
                      size="0.165 0.165 0.001"
                      pos="0.164 .164 -0.01"
                      rgba="0.1 .1 0 .5"/> -->

                <geom type="mesh"
                      mesh="fdm_printer_bed"
                      material="black_metal"
                      friction="0.4 0.4 0.8"/>


            </body>
        </body>
    </worldbody>
    <actuator>
        <position name="ax" gear="100" joint="xaxis" forcelimited="true" forcerange="-10000 10000" kp='100'/>
        <position name="ay" gear="100" joint="yaxis" forcelimited="true" forcerange="-10000 10000" kp='100'/>
        <position name="az" gear="100" joint="zaxis" forcelimited="true" forcerange="-10000 10000" kp='100'/>
    </actuator>
    <sensor>
        <actuatorpos name="x" actuator="ax"/>
        <actuatorpos name="y" actuator="ay"/>
        <actuatorpos name="z" actuator="az"/>
    </sensor>
</mujoco>
    """
)
class AnetA30:

    def __init__(self):
        pass

    def home(self):
        # implemented by the interface
        pass

    def is_at(self, position):
        # implemented by the interface
        pass

    def move(self, position: [float, float, float]):
        # in millimeters (mm)
        # implemented by the interface
        pass
