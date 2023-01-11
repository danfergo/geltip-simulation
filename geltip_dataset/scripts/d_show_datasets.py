import os
import cv2

from experimental_setup.geltip.sim_model.scripts.utils.vis import show_panel


def main():
    dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'

    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
    fields = ['linear', 'geodesic', 'combined',
              'linear_elastic', 'geodesic_elastic', 'combined_elastic',
              'linear_bkg', 'geodesic_bkg', 'combined_bkg',
              'linear_elastic_bkg', 'geodesic_elastic_bkg', 'combined_elastic_bkg']

    N_ROWS = 3
    N_CONTACTS = 6

    # show
    for obj in objects:
        for i in range(N_ROWS):
            for j in range(N_CONTACTS):
                imgs = [cv2.cvtColor(cv2.imread(dataset_path + 'sim_' + field + '/' + obj + '/' + str(i) + '_' + str(j) + '.png'), cv2.COLOR_BGR2RGB)
                        for field in fields]

                show_panel(imgs, (4, len(fields) // 4))


if __name__ == '__main__':
    main()
