import yaml

import numpy as np

# --- localisation
# ex = 'train_localizer'
# run = '2023-02-03 15:15:01'  # localisation sim2sim

# --- classification
ex = 'train_classifier'

# run = '2023-02-05 14:49:13' # classificatin real2real
# run = '2023-02-05 17:09:27' # classificatin sim2sim
run = '2023-02-05 18:12:26'  # classification sim2sim cropped
run = '2023-02-05 18:12:26'  # classification sim2sim cropped

file_path = f'./outputs/{ex}/runs/{run}/out/stats.yaml'

data = yaml.full_load(open(file_path))

val_data = data['val']
train_data = data['train']

best_it = np.argmin(val_data['all_losses'])

print('-----------------')
print('best loss:')
print('train: ' + str(val_data['all_losses'][best_it]))
print('val: ' + str(train_data['all_losses'][best_it]))

print('-----------------')
print('best metrics:')

for i in range(len(val_data['all_metrics'])):
    print('train: ' + str(train_data['all_metrics'][i][best_it]))
    print('val: ' + str(val_data['all_metrics'][i][best_it]))
