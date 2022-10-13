from train import *

import requests
import json
import time
import numpy as np
import traceback


with open('search-server-ip', 'r') as f:
    url = f.read().strip()
p_data = None

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    while True:
        try:
            r = requests.get(url + 'get_eval')
            if r.status_code == 200:
                break
            time.sleep(10)
        except ConnectionRefusedError:
            print("Connection Refused Error, retry in 10s.")

    p_data = json.loads(r.text)
    print(p_data)

    data = p_data['data']
    aug = data[0]
    rl_aug = data[1]
    weights = data[2:]
    print(f"Aug: {aug} | RL_Aug: {rl_aug} | Weights: ", weights)

    main(weights, aug=aug, rl_aug=rl_aug, task_str=r.text)

    print("Done")


