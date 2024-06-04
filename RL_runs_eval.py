import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)


run_no_l = [1,2,3,4]
hidden_layers_l = [1]
neuron_num_l = [64, 128, 256, 512]
plt.figure(200)
plt.clf()
max_rewards = []
for j in range(len(hidden_layers_l)):
    for i in range(len(run_no_l)):
        run_no = run_no_l[i]
        hidden_layers = hidden_layers_l[j]
        neuron_num = neuron_num_l[i]
        # file_name = 'Files/' + str(run_no) +'_' + str(hidden_layers) +\
        #     '_' + str(neuron_num) + '_rewards.pickle'
        file_name = 'Files/' + str(hidden_layers) + \
                    '_' + str(neuron_num) + '_cool2_rewards.pickle' #cool2_rewards volt _cool_no_ek_rewards.pickle

        window = 50
        if hidden_layers == 1:
            legend_lab = str(hidden_layers) + ' hidden_layer,' + str(neuron_num) + ' neurons'
        else:
            legend_lab = str(hidden_layers) + ' hidden_layer,' + str(neuron_num) + 'x' + str(neuron_num) + ' neurons'

        with open(file_name, 'rb') as f:
            episode_rewards = pickle.load(f)


        moving_avg = np.convolve(episode_rewards, np.ones((window,)) / window,
                                         mode='valid')

        max_rewards.append(max(moving_avg))

        plt.plot(moving_avg, label=legend_lab)

plt.xlabel('Episode', fontsize=font['size'])
plt.ylabel('Average reward', fontsize=font['size'])
plt.legend()
plt.xlim([0, 10_000])