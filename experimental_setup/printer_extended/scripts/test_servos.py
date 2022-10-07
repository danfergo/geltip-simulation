import os

from yarok import PlatformMJC, PlatformHW, ConfigBlock
from yarok.components_manager import component
import yarok

from math import pi, cos, sin
import cv2
import numpy as np
from yarok.components.geltip.geltip import GelTip

from experimental_setup.printer_extended.printer_extended import PrinterExtended


@component(
    components=[
        PrinterExtended,
        GelTip
    ],
    template="""
        <mujoco>
            <compiler angle="radian"/>

            <visual>
                <quality shadowsize="2048"/>
            </visual>

            <asset>
                <!-- empty world -->
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                         width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>    
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            </asset>

            <worldbody>
                <light directional="true" diffuse=".9 .9 .9" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1"/>
                <camera name="viewer" pos="0 0 0.5" mode="fixed" zaxis="0 0 1"/>

                <body name="floor">
                    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                </body>

                <printer_extended name="printer"></printer_extended>

            </worldbody>    
        </mujoco>
    """
)
class TestServosWorld:
    pass


class TestServosBehaviour:

    def __init__(self, printer: PrinterExtended, config: ConfigBlock):
        self.config = config
        self.printer = printer

        self.GAMMA_AMPLITUDE = 180  # total rotation amplitude to probe
        self.R_STEPS = 3

        # the rotation incremental amplitude
        self.dGamma = self.GAMMA_AMPLITUDE / (self.R_STEPS - 1) \
            if self.R_STEPS > 1 else self.GAMMA_AMPLITUDE

    def move_servos(self, angles=None):
        print('move: ', angles)
        self.printer.move_servos(angles)
        yarok.wait(lambda: self.printer.servos_at(angles))
        x = input("press any key: \n")

    def wake_up(self):
        for r in range(self.R_STEPS):
            print('Rotating Servo 2, i.e. the rotating the sensor')
            gamma = round(r * self.dGamma)
            self.move_servos((0, gamma))


if __name__ == '__main__':
    # objects = ['cone', 'cylinder', 'cylinder_shell', 'dot_in', 'dots', 'pacman', 'sphere', 'random']
    objects = ['cone']
    env = 'real'
    save_data = True

    for obj in objects:
        yarok.run({
            'world': TestServosWorld,
            'behaviour': TestServosBehaviour,
            'defaults': {
                'environment': env,
            },
            'environments': {
                'sim': {
                    'platform': {
                        'class': PlatformMJC
                    },
                    'inspector': False,
                },
                'real': {
                    'platform': PlatformHW,
                    'behaviour': {
                        'dataset_name': 'real_rgb'
                    },
                    'interfaces': {
                        '/printer/printer': {
                            'serial_path': '/dev/ttyUSB0'
                        },
                        '/printer': {
                            'serial_path': '/dev/ttyUSB1'
                        },
                    }
                }
            },
        })
