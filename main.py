import tensorflow as tf
import os
from variant import VARIANT, get_env_from_name,  get_train, get_eval
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    root_dir = VARIANT['log_path']
    if VARIANT['train']:
        for i in range(VARIANT['start_of_trial'], VARIANT['start_of_trial']+VARIANT['num_of_trials']):
            VARIANT['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + VARIANT['log_path'])
            train = get_train(VARIANT['algorithm_name'])
            train(VARIANT)

            tf.reset_default_graph()
    else:
        eval = get_eval(VARIANT['algorithm_name'])
        eval(VARIANT)

