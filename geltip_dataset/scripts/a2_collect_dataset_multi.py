import os

from yarok import Platform, PlatformMJC, PlatformHW, ConfigBlock, Injector, component

from math import pi, cos, sin
from experimental_setup.geltip.geltip import GelTip

from experimental_setup.printer_extended.printer_extended import PrinterExtended

import numpy as np
import cv2


@component(
    components=[
        PrinterExtended,
        GelTip
    ],
    defaults={
        'n_printers': 20,
        'row_size': 5
    },
    template="""
        <mujoco>
            <compiler angle="radian"/>

            <visual>
                <!-- important for the Geltips, to ensure the its camera frustum captures the close-up elastomer -->
                <map znear="0.001" zfar="50"/>
                <quality shadowsize="2048"/>
            </visual>

            <asset>
                <!-- empty world -->
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                         width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>    
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>

                <!-- object set -->
                <material name="black_plastic" rgba=".3 .3 .3 1"/>
                <mesh name="object" file="../object_set/${object}.stl" scale="0.001 0.001 0.001"/>
            </asset>

            <worldbody>
                <light directional="true" diffuse=".9 .9 .9" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1"/>
                <camera name="viewer" pos="0 0 0.5" mode="fixed" zaxis="0 0 1"/>

                <body name="floor">
                    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                </body>

                <for each="range(n_printers)" as="i">
                    <body pos="${i % row_size} ${i//row_size} 0">
                        <printer_extended name="printer${i}">
                            <geom 
                                 parent="indenter_mount"
                                 pos="0.001 0.001 0.001"
                                 type="mesh"
                                 density="0.1"
                                 mesh="object"
                                 material="black_plastic"/>
                             <geltip name="geltip${i}" parent="geltip_mount"/>
                        </printer_extended>
                    </body>
                </for>
            </worldbody>    
        </mujoco>
    """
)
class DatasetCollectionWorld:

    def __init__(self, config: ConfigBlock):
        self.n_printers = config['n_printers']


class DatasetCollectionBehaviour:

    def __init__(self, injector: Injector, config: ConfigBlock, pl: Platform):

        self.pl = pl
        self.config = config
        self.n_printers = injector.get('world').n_printers
        self.printers = [injector.get('printer' + str(i)) for i in range(self.n_printers)]
        self.geltips = [injector.get('geltip' + str(i)) for i in range(self.n_printers)]
        self.data_path = config['dataset_path'] + config['dataset_name']
        self.object_name = config['object']

        self.H_LENGTH = 18
        self.RADIUS = 11  # the geltip v1 elastomer radius is 11.
        self.HEIGHT = self.H_LENGTH + self.RADIUS

        self.ROT_STEPS = 8  # tip rotations
        self.TRA_STEPS = 9  # the first translation step is skipped,
        # as it is shared with the rotations
        # total number of contacts = ROT + TRA - 1, 0-indexed

        self.N_ROWS = 6  # longitudinal rows/lines/paths
        self.GAMMA_AMPLITUDE = 180  # angle rotation between rows/lines/paths

        self.INDENTER_H = 20  # 10mm base, 10mm indenter per say

        self.GELTIP_T = 260, 204, 29  # geltip tip, i.e. the top, center
        self.GELTIP_C = self.GELTIP_T[0] - self.RADIUS - self.INDENTER_H, \
                        self.GELTIP_T[1], \
                        self.GELTIP_T[2]  # geltip center i.e. center of the curved tip

        SAFE_Z = 93  # safe height for the printer head to traverse the printer bed without
        # colliding against the sensor and sensor mounts
        # Some preparation points that the printer traverses to reach near the sensor
        # without colliding against it
        self.prepare_pts = [(0, 0, SAFE_Z), (self.GELTIP_T[0], self.GELTIP_T[1], SAFE_Z)]

        # the rotation incremental amplitude
        self.dGamma = self.GAMMA_AMPLITUDE / (self.N_ROWS - 1) \
            if self.N_ROWS > 1 else self.GAMMA_AMPLITUDE

        # margin above the sensor surface, for moving the sensor from contact to contact
        self.MARGIN = 10
        # how many mm the indenter protrudes the elastomer surface
        self.PROTRUSION_DEPTH = 2
        self.LOW_PATH = self.RADIUS + self.INDENTER_H - self.PROTRUSION_DEPTH
        self.HIGH_PATH = self.RADIUS + self.INDENTER_H + self.MARGIN

    def save_frame(self, key):
        if not self.config['save_data']:
            return

        # frame_path = self.data_path + '/' + self.object_name + '/' + key
        #
        # if self.config['dataset_name'].split('_')[0] == 'sim':
        #     def save_np(p, d):
        #         with open(p + '.npy', 'wb') as f:
        #             np.save(f, d)
        #
        #     [save_np(frame_path + '__' + str(i), self.geltips[i].read_depth()) for i in range(self.n_printers)]
        # else:
        #     [cv2.imwrite(frame_path + '__' + str(i) + '.png', self.geltips[i].read()) for i in range(self.n_printers)]
        # print('saved ' + key)

    def save_data_frame(self, r, ith):
        self.save_frame(str(r) + '_' + str(ith))

    def move(self, position=None, angles=None):
        print('move: ', position, angles)
        self.pl.wait([printer.move(position, angles) for printer in self.printers])
        # self.pl.wait(lambda: not any([not printer.is_at(position, angles) for printer in self.printers]))
        # t = time.time()
        # self.pl.wait_seconds(100)
        # x = input("press any key: \n")

    def movec(self, position=None, angles=None):
        p = None if position is None else \
            tuple([self.GELTIP_C[i] + position[i] for i in range(3)])
        self.move(p, angles)

    def on_start(self):

        # move from the 0,0 / home position to a higher position,
        # so that it doesnt collide with clamps and the sensor
        # when moving to the first position
        [self.move(p) for p in self.prepare_pts]
        print('Ended preparation points. ')

        # collect a background fame for reference
        self.save_frame('bkg')

        for r in range(self.N_ROWS):

            # move to the starting position,
            # make sure that the sensor is well aligned
            self.move((self.GELTIP_T[0], self.GELTIP_T[1], 66))
            # self.move(self.GELTIP_T, (90, 0))

            # rotate sensor, to collect corresponding row
            print('Rotating the geltip sensor, to collect a new row')
            gamma = round(r * self.dGamma)
            self.move(angles=(0, gamma))

            r_low = self.LOW_PATH
            r_high = self.HIGH_PATH
            dt = (pi / 2) / (self.ROT_STEPS - 1)
            d0x = lambda theta: cos(theta * dt)
            d0y = lambda theta: sin(theta * dt)
            rad2deg = lambda radians: round(radians * 180 / pi)

            # trace curved path
            for rot in range(0, self.ROT_STEPS):
                theta = 90 - rad2deg(rot * dt)

                self.movec((r_high * d0x(rot), 0, r_high * d0y(rot)),
                           (theta, gamma))  # up

                self.movec((r_low * d0x(rot), 0, r_low * d0y(rot)),
                           (theta, gamma))  # down

                self.save_data_frame(r, rot)

                self.movec((r_high * d0x(rot), 0, r_high * d0y(rot)),
                           (theta, gamma))  # up

            # if running after horizontal path,
            # it should result in no movement.
            # self.movec((0, 0, self.HIGH_PATH), (0, 0))

            # trace linear path
            dx = self.H_LENGTH / (self.TRA_STEPS - 1)
            for tra in range(1, self.TRA_STEPS):
                self.movec((- tra * dx, 0, self.HIGH_PATH))

                # capture frame.
                self.movec((- tra * dx, 0, self.LOW_PATH))

                self.save_data_frame(r, self.TRA_STEPS + tra)
                self.movec((- tra * dx, 0, self.HIGH_PATH))

            print('ended.')

            # mv to safe point.
            # self.printer.move((p[0] + r_high, p[1], self.pts[0][2]), (0, gamma))
            # yarok.wait(lambda: self.printer.is_at((p[0] + r_high, p[1], self.pts[0][2]), (0, gamma)))

            # self.printer.move(self.pts[0])
            # yarok.wait(lambda: self.pts[0])


def main():
    """
        It would be more elegant to have a Behaviour being associated to each printer,
        rather than a global Behaviour controlling all printers,
        But Yarok didn't support it when implementing this data collection.
    """
    objects = [
        'cone',
        # 'sphere',
        # 'cylinder_shell', 'dot_in',
        # 'dots', 'pacman',
        # 'sphere', 'random'
    ]
    n_printers = 20
    for obj in objects:
        Platform.create({
            'world': DatasetCollectionWorld,
            'behaviour': DatasetCollectionBehaviour,
            'defaults': {
                'environment': 'sim',
                'behaviour': {
                    'dataset_path': os.path.dirname(os.path.abspath(__file__)) + '/../dataset/',
                    'object': obj,
                    'save_data': True
                },
                'components': {
                    '/': {
                        'n_printers': n_printers,
                        'object': obj
                    },
                    **{
                        '/geltip' + str(i): {
                            'label_color': f'{i / n_printers} {i / n_printers} {i / n_printers}'
                        }
                        for i in range(n_printers)
                    }
                },
                'plugins': []
            },
            'environments': {
                'sim': {
                    'platform': {
                        'class': PlatformMJC
                    },
                    'behaviour': {
                        'dataset_name': 'sim_adepth'
                    }
                }
            },
        }).run()


if __name__ == '__main__':
    main()
