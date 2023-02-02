import cv2
from yarok import Platform, PlatformMJC, ConfigBlock, Injector, component

import os
import numpy as np

from experimental_setup.geltip.geltip import GelTip
from sim_model.utils.vis_img import to_normed_rgb, to_panel


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
                
                <!-- to test camera left-right -->
                <!-- 
                <body>
                    <geom type='box'
                        pos='0.1 0.1 .1'
                        rgba='255 0 0 1'
                        size='0.02 0.02 0.02'/>
                </body> -->

                <body pos="0.0 0.0 0.1" xyaxes='-1 0 0 0 0 1'>
                    <geltip name="geltip1"/>
                </body>

                <body pos="0.05 0.0 0.1" xyaxes='-1 0 0 0 0 1'> 
                    <geltip name="geltip2"/>
                    <body>
                         <geom pos=".0095 -.0075 .04" size=".0035" rgba="0 1 1 1"/>
                    </body>
                </body>

                <body pos="0.1 0.0 0.1"  xyaxes='-1 0 0 0 0 1'> 
                    <geltip name="geltip3"/>
                    <body>
                         <geom pos="-.01 -.009 .04" size=".0055" rgba="1 0 0 1"/>
                    </body>
                </body>

                <body pos="0.15 0.0 0.1" xyaxes='-1 0 0 0 0 1'> 
                    <geltip name="geltip4"/>
                    <body>
                         <geom pos=".009 .009 .04" size=".0065" rgba="1 0 0 1"/>
                    </body>
                </body>
                
                <body pos="0.2 0.0 0.1" xyaxes='-1 0 0 0 0 1'> 
                    <geltip name="geltip5"/>
                    <body>
                         <geom pos=".0 .0 .05" size=".0065" rgba="1 0 0 1"/>
                    </body>
                </body>
                
                <body pos="0.25 0.0 0.1" xyaxes='-1 0 0 0 0 1'> 
                    <geltip name="geltip6"/>
                    <body>
                         <geom pos=".004 .0 .05" size=".0065" rgba="1 0 0 1"/>
                    </body>
                </body>

            </worldbody>        
        </mujoco>
    """
)
class GelTipWorld:
    pass


class CaptureDepthSampleBehaviour:

    def __init__(self,
                 injector: Injector,
                 config: ConfigBlock,
                 pl: Platform):
        self.sensors = [injector.get('geltip' + str(i)) for i in range(1, 7)]
        self.config = config
        self.pl = pl

    def save_depth_frame(self, geltip, key):
        frame_path = self.config['assets_path'] + '/' + key
        with open(frame_path + '.npy', 'wb') as f:
            depth_frame = geltip.read_depth()
            np.save(f, depth_frame)
            return depth_frame

    def on_start(self):
        self.pl.wait_seconds(5)

        frames = [
            to_normed_rgb(self.save_depth_frame(g, 'bkg' if i == 0 else 'depth_' + str(i)))
            for i, g in enumerate(self.sensors)
        ]

        cv2.imshow('frames', to_panel(frames, shape=(2, 3)))
        cv2.waitKey(-1)


if __name__ == '__main__':
    __location__ = os.path.dirname(os.path.abspath(__file__))
    Platform.create({
        'world': GelTipWorld,
        'behaviour': CaptureDepthSampleBehaviour,
        'defaults': {
            'environment': 'sim',
            'behaviour': {
                'assets_path': os.path.join(__location__, '../../experimental_setup/geltip/sim_assets')
            },
            'components': {
                '/': {
                    'object': 'cone'
                },
                '/geltip1': {'label_color': '0 0 1'},
                '/geltip2': {'label_color': '0 1 0'},
                '/geltip3': {'label_color': '1 0 0'},
                '/geltip4': {'label_color': '1 1 0'},
                '/geltip5': {'label_color': '0 1 1'},
                '/geltip6': {'label_color': '1 0 1'}
            },
            'plugins': [
                # (Cv2Inspector, {})
            ]
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC
                },
                'inspector': False
            },
        },
    }).run()
