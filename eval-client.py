from train import *

import requests
import json
import time
import numpy as np
import traceback


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    for i in range(1, 21):
        js_file = f'server/top{i}.json'
        print(js_file)
        with open(js_file, 'r') as f:
            p_data = json.load(f)
        print(p_data)

        data = p_data['data']
        aug = data[0]
        rl_aug = data[1]
        weights = data[2:]
        print(f"Aug: {aug} | RL_Aug: {rl_aug} | Weights: ", weights)

        main(weights, aug=aug, rl_aug=rl_aug, task_str=json.dumps(p_data), tid=i)

        print(f"{js_file} done!")


