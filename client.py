from train import *

import requests
import json
import time
import numpy as np
import traceback
import os

with open('server-addr', 'r') as f:
    url = f.readline().strip()
    print("URL: " + url)
p_data = None


def check_should_run():
    return not os.path.exists('stop')

if __name__ == '__main__':
    envs = [
        ('cartpole', 'swingup', 8),
        ('cheetah', 'run', 4),
        ('reacher', 'easy', 4),
        ('ball_in_cup', 'catch', 4),
        ('finger', 'spin', 2),
        ('walker', 'walk', 2)
    ]

    torch.multiprocessing.set_start_method('spawn')
    is_running = check_should_run()

    while is_running:
        is_running = check_should_run()
        while is_running:
            try:
                r = requests.get(url + 'get_task')
                if r.status_code == 200:
                    break
                time.sleep(10)
                is_running = check_should_run()
            except ConnectionRefusedError:
                print("Connection Refused Error, retry in 10s.")

        if not is_running:
            break

        p_data = json.loads(r.text)
        print(p_data)
        if 'stop' in p_data:
            is_running = False
            break

        data = p_data['data']
        aug = data[0]
        rl_aug = data[1]
        weights = data[2:]
        print(f"Aug: {aug} | RL_Aug: {rl_aug} | Weights: ", weights)

        try:
            results = {}
            for env in envs:
                domain, task, action_repeat = env
                results[domain + ' ' + task] = [main(weights,
                                domain=domain,
                                task=task,
                                action_repeat=action_repeat,
                                tid=p_data['tid'], aug=aug, rl_aug=rl_aug, seed=i, task_str=r.text) for i in p_data['seed']]
            p_data['results'] = results
            p_data['status'] = 0
        except KeyboardInterrupt:
            p_data['status'] = 1
            is_running = False
        except:
            traceback.print_exc()
            p_data['status'] = 2
            time.sleep(10)
        while True:
            try:
                r = requests.post(url + 'submit_result', data = json.dumps(p_data))
                if r.status_code == 200:
                    break
                time.sleep(10)
            except ConnectionRefusedError:
                print("Connection Refused Error, retry in 10s.")

        print("Done")

