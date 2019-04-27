import glob
import os

import numpy as np
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing
import matplotlib.pyplot as plt
from gym.envs.box2d import car_racing
import cv2

CROP_SIZE = 84
CROP_W_OFFSET = int((car_racing.STATE_W - CROP_SIZE)/2)


def normalize_observation(observation):
    return observation.astype(np.float32) / 255.


def main():
    print("Generating data for env CarRacing-v0")

    env = CarRacing()

    for obs_idx in range(2, 10):

        env.reset()

        observations = []

        for i in range(1000):
            position = np.random.randint(len(env.track))
            angle = np.random.randint(-20, 20)
            x_off = np.random.randint(-20, 20)
            init_data = list(env.track[position][1:4])
            init_data[0] += angle
            init_data[1] += x_off
            env.car = Car(env.world, *init_data)

            observation = env.step(None)[0]

            cropped_obs = normalize_observation(observation[:CROP_SIZE, CROP_W_OFFSET:CROP_SIZE+CROP_W_OFFSET, :])

            cropped_obs = cv2.resize(cropped_obs, dsize=(64, 64), interpolation=cv2.INTER_CUBIC).astype(np.float32)

            if i % 10 == 0:
                print(i)

            if i % 100 == 0:
                plt.imshow(cropped_obs)
                plt.show()

            observations.append(cropped_obs)

        observations = np.array(observations, dtype=np.float32)

        if not os.path.exists("data"):
            os.mkdir("data")

        np.save("data/observations_%d.npy" % obs_idx, observations)


def combine_observations():
    observations = []
    data_files = glob.glob('data/observations_*.npy')
    for d in data_files:
        obs_data = np.load(d)
        observations.extend(obs_data)

    observations = np.array(observations, dtype=np.float32)

    np.save("data/observations.npy", observations)

    print(data_files)


if __name__ == "__main__":
    #main()
    combine_observations()