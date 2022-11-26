"""
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eOWP8Q_3zVOyjqLuWWmNqdBlpU-ZH6_v' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eOWP8Q_3zVOyjqLuWWmNqdBlpU-ZH6_v" -O 1Deg_800Sample.mat && rm -rf /tmp/cookies.txt


"""
# print("X")

import mat73

# TODO: make rainymotion import work
# from rainymotion.models import Dense

import pysteps

# print("Y")

#%% load the data
mat = mat73.loadmat('../../data/1Deg_800Sample.mat')  # 8 time step estimation
X_1 = mat[
    'X_train']  # (sample, time sequence, latitude, longitude, channel) here channels are 1: precipitation, 2: wind velocity in x direction, 3: wind velocity in y direction
y_1 = mat['y_train']  # (sample, time sequence, lat, lon)

# print("Z")

X_test = mat['X_test']
y_test = mat['y_test']
GFS = mat['GFS_test']

print("[X] All data loaded")

# # TODO: use pysteps model and save output
# print(f"X_train: \n\t{X_1.shape}")
# print(f"y_train: \n\t{y_1.shape}")
# print(f"X_test: \n\t{X_test.shape}")
# print(f"y_test: \n\t{y_test.shape}")


