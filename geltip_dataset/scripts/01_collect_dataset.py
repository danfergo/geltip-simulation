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

                <printer_extended name="printer">
                    <geom 
                         parent="indenter_mount"
                         type="mesh"
                         density="0.1"
                         mesh="object"
                         material="black_plastic"/>
                    <geltip name="geltip" parent="geltip_mount"/>
                </printer_extended>

            </worldbody>    
        </mujoco>
    """
)
class DatasetCollectionWorld:
    pass


class DatasetCollectionBehaviour:

    def __init__(self, printer: PrinterExtended, geltip: GelTip, config: ConfigBlock):
        self.config = config
        self.printer = printer
        self.geltip = geltip
        self.data_path = config['dataset_path'] + config['dataset_name']
        self.object_name = config['object']

        self.H_LENGTH = 18
        self.RADIUS = 11  # the geltip v1 elastomer radius is 11.
        self.HEIGHT = self.H_LENGTH + self.RADIUS
        self.ROT_STEPS = 4  # tip rotations
        self.TRA_STEPS = 3  # the first translation step is skipped,
        # as it is shared with the rotations
        # total number of contacts = ROT + TRA - 1, 0-indexed
        self.N_ROWS = 3  # rows
        self.GAMMA_AMPLITUDE = 180  # total rotation amplitude to probe

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
        # if self.config['dataset_name'] == 'sim_depth':
        #     with open(frame_path + '.npy', 'wb') as f:
        #         depth_frame = self.geltip.read_depth()
        #         print('min', depth_frame.min(), 'max', depth_frame.max())
        #         np.save(f, depth_frame)
        # else:
        #     cv2.imwrite(frame_path + '.png', self.geltip.read())

    def save_data_frame(self, r, ith):
        self.save_frame(str(r) + '_' + str(ith))

    def move(self, position=None, angles=None):
        print('move: ', position, angles)
        self.printer.move(position, angles)
        yarok.wait(lambda: self.printer.is_at(position, angles))
        # t = time.time()
        # yarok.wait(lambda: (time.time() - t) > 2)
        # x = input("press any key: \n")

    def movec(self, position=None, angles=None):
        p = None if position is None else \
            tuple([self.GELTIP_C[i] + position[i] for i in range(3)])
        self.move(p, angles)

    def on_update(self):

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


if __name__ == '__main__':
    objects = [
        # 'cone',
        'sphere',
        # 'cylinder_shell', 'dot_in',
        # 'dots', 'pacman',
        # 'sphere', 'random'
    ]
    # objects = ['cone']
    # objects = ['cone']
    env = 'sim'
    save_data = False

    for obj in objects:
        yarok.run({
            'world': DatasetCollectionWorld,
            'behaviour': DatasetCollectionBehaviour,
            'defaults': {
                'environment': env,
                'behaviour': {
                    'dataset_path': os.path.dirname(os.path.abspath(__file__)) + '/../dataset/',
                    'object': obj,
                    'save_data': save_data
                },
                'components': {
                    '/': {
                        'object': obj
                    },
                    '/geltip': {
                        'label_color': '.5 .5 .5'
                    }
                }
            },
            'environments': {
                'sim': {
                    'platform': {
                        'class': PlatformMJC
                    },
                    'inspector': False,
                    'behaviour': {
                        'dataset_name': 'sim_depth'
                    }
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
                        '/geltip': {
                            'cam_id': 2
                        }
                    }
                }
            },
        })
