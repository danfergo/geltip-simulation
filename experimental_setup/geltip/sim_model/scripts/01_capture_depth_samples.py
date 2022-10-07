from yarok import PlatformMJC, ConfigBlock
import yarok

import cv2
import numpy as np
import os

from yarok.components_manager import component

from experimental_setup.geltip.geltip import GelTip
from experimental_setup.geltip.sim_model.scripts.utils.vis import to_normed_rgb, to_panel


@component(
    components=[
        GelTip
    ],
    # language=xml
    template="""
        <mujoco>
            <visual>
                <!-- important for the GelTips, to ensure its camera frustum captures the close-up elastomer -->
                <map znear="0.001" zfar="50"/>
            </visual>

            <asset>
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                         width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>    
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            </asset>
            <worldbody>
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1"/>
                <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>

                <body name="floor">
                    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                </body>

                <body pos="0.0 0.0 0.1">
                    <geltip name="geltip1"/>
                </body>


                <body pos="0.05 0.05 0.1"> 
                    <geltip name="geltip2"/>
                    <body>
                        <!-- <geom pos=".009 -.009 .045" size=".0035" rgba="0 1 0 1"/> -->
                         <geom pos=".009 -.007 .04" size=".0035" rgba="0 1 1 1"/>
                    </body>
                </body>

                <body pos="0.1 0.1 0.1"> 
                    <geltip name="geltip3"/>
                    <body>
                        <!-- <geom pos=".009 -.009 .045" size=".0035" rgba="0 1 0 1"/> -->
                         <geom pos="-.009 -.008 .04" size=".0055" rgba="1 0 0 1"/>
                    </body>
                </body>

                <body pos="0.3 0.3 0.1"> 
                    <geltip name="geltip4"/>
                    <body>
                        <!-- <geom pos=".009 -.009 .045" size=".0035" rgba="0 1 0 1"/> -->
                         <geom pos=".009 -.009 .04" size=".0065" rgba="1 0 0 1"/>
                    </body>
                </body>

            </worldbody>        
        </mujoco>
    """
)
class GelTipWorld:

    def __init__(self, geltip1: GelTip):
        self.geltip1 = geltip1


class CaptureDepthSampleBehaviour:

    def __init__(self,
                 geltip1: GelTip,
                 geltip2: GelTip,
                 geltip3: GelTip,
                 geltip4: GelTip,
                 config: ConfigBlock):
        self.config = config
        self.geltips = [geltip1, geltip2, geltip3, geltip4]
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

    def save_depth_frame(self, geltip, key):
        frame_path = self.data_path + '/' + key
        with open(frame_path + '.npy', 'wb') as f:
            depth_frame = geltip.read_depth()
            np.save(f, depth_frame)
            return depth_frame

    def wake_up(self):
        yarok.wait_seconds(5)

        frames = [
            to_normed_rgb(self.save_depth_frame(g, 'bkg' if i == 0 else 'depth_' + str(i)))
            for i, g in enumerate(self.geltips)
        ]

        # cv2.imshow('frames', to_panel(frames))


if __name__ == '__main__':
    yarok.run({
        'world': GelTipWorld,
        'behaviour': CaptureDepthSampleBehaviour,
        'defaults': {
            'environment': 'sim',
            'behaviour': {
                'dataset_path': '/home/danfergo/Projects/PhD/geltip_simulation/dataset/'
            },
            'components': {
                '/': {
                    'object': 'cone'
                },
                '/geltip1': {'label_color': '0 0 1'},
                '/geltip2': {'label_color': '0 1 0'},
                '/geltip3': {'label_color': '1 0 0'},
                '/geltip4': {'label_color': '1 0 0'}
            }
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC
                },
                'inspector': False
            },
        },
    })
