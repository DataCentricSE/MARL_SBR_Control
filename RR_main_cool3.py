##

from importlib import reload
import Modell_reactor
reload(Modell_reactor)
from Modell_reactor import Reactor
import params
reload(params)

from params import REACTOR_PARAMS, INIT_PARAMS, MAIN_PARAMS, OP_PARAMS
import matplotlib.pyplot as plt
from DDPG_agent import Agent
import numpy as np
from datetime import datetime
from PID_controller import PID
import pickle


MAIN_PARAMS["SHOW_EVERY"] = 50

episodes = 10_000
observation_space_dim_feed = 8
observation_space_dim_cool = 7

agent_feed = Agent(alpha=0.0001, beta=0.001, input_dims=[observation_space_dim_feed],
                  tau=0.001, batch_size=64, max_size = 1_000_000, hidden_layers=2, action_interval=10, fc1_dims=128, fc2_dims=128,
                  n_actions=1, name='2_128') #ha hidden_layers=1, akkor fc2_dims nem számít

neuronnum = 512
hl = 1
fignum = neuronnum
agent_cool = Agent(alpha=0.0001, beta=0.001, input_dims=[observation_space_dim_cool],
                  tau=0.001, batch_size=64, max_size = 1_000_000, hidden_layers=hl, action_interval=10,
                   fc1_dims=neuronnum, fc2_dims=neuronnum,
                  n_actions=1, name=str(hl) + '_' + str(neuronnum) + '_cool_no_ek') #ha hidden_layers=1, akkor fc2_dims nem számít


episode_rewards = []
reactor = Reactor(REACTOR_PARAMS, INIT_PARAMS)
agent_feed.load_models()

#PID controller
PID_master = PID(K=5, TI=200)   #IMC-ből: K=4, TI = 307.5 (tauc=tau/4), vagy K=5, TI=246 (tauc=tau/2)
PID_slave = PID(K=14.8, TI=12.4) #IMC-ből: K=14.8, TI=12,4
PID_mode = 1 #0: Nem kaszkád, TR_OP_c, 1: kaszkád
Master_on = 1 #1 - ON
d_lim = 10 #OP_c max váltás
open_loop_on = 0 #Slave, 1 - on, 0 - off
open_loop_on_master = 0
t_openloop = 0

SP = 363 #K
best_ag_r = 0
start_time = datetime.now()
for episode in range(episodes):
    error_master = [0, 0]
    error_slave = [0, 0]
    ep_start_time = datetime.now()
    reactor.reset()
    br = 0
    cr = 0
    episode_reward = 0

    reactor.TR = np.random.randint(333,364)
    reactor.n_B = np.random.uniform(4,5)
    OP_PARAMS["nA_feed"] = reactor.n_B

    PV = reactor.TR
    OP = 0
    c_A = reactor.n_A/reactor.V
    c_B = reactor.n_B/reactor.V
    OP_c = 100
    UA = reactor.UA

    traj = [[reactor.time], [SP], [PV], [OP], [reactor.TJ], [reactor.n_A], [reactor.n_B], [OP_c], [0], [0], [reactor.TJ]]
    SP_slave = reactor.TJ

    PID_traj = [[reactor.time], [SP], [PV], [reactor.TJ], [reactor.TJ], [OP_c]]
    obs_feed = (PV, OP, reactor.TJ, c_A, c_B, UA, error_master[0], error_master[1])

    # obs_cool = (PV, OP, SP_slave, reactor.TJ, c_A, c_B, UA, error_master[0], error_master[1])
    obs_cool = (PV, OP, SP_slave, reactor.TJ, c_A, c_B, UA)
    # obs_cool = (error_master[0], error_master[1])
    if episode % MAIN_PARAMS["SHOW_EVERY"] == 0:
        print(f"{MAIN_PARAMS['SHOW_EVERY']} ep mean: {np.mean(episode_rewards[-MAIN_PARAMS['SHOW_EVERY']:])}")
        if episode % 100 == 0:
            show = True
        else:
            show=False
    else:
        show = False



    for i in range(int(4*3600/MAIN_PARAMS["OP_H"])):
        if reactor.op_mode == 0:
            rl_sampl = 100
        else:
            rl_sampl = 100

        if reactor.op_mode == 1 and cr == 1:
            PID_traj[0][0] = reactor.time
            PID_traj[2][0] = PV
            PID_traj[3][0] = SP_slave
            PID_traj[4][0] = reactor.TJ
            PID_traj[5][0] = OP_c
            cr += 1

        if reactor.op_mode == 0:    #RL - FEED VALVE control
            if int(reactor.time) % 100 == 0:
                action = agent_feed.choose_action(obs_feed)
                OP_b = OP
                OP += float(action)
                OP = 0 if OP < 0 else OP
                OP = 100 if OP > 100 else OP
                OP_c = 100
        else:                      #PID - COOLING AGENT VALVE control

            #Master köri szabályzás - RL:
            if int(reactor.time) % rl_sampl == 0:
                action = agent_cool.choose_action(obs_cool)
                SP_slave += action
                SP_slave = 370 if SP_slave>370 else SP_slave
                SP_slave = 298 if SP_slave < 298 else SP_slave


            #Slave köri szabályzás - PID:
            error_slave[0] = error_slave[1]
            error_slave[1] = -(SP_slave - reactor.TJ)
            action_slave = PID_slave.update(error_slave[-2:])

            action_slave = d_lim if action_slave>d_lim else action_slave
            action_slave = -d_lim if action_slave<-d_lim else action_slave
            OP_c += float(action_slave)
            OP_c = 0 if OP_c<0 else OP_c
            OP_c = 100 if OP_c>100 else OP_c

        for _ in range(int(MAIN_PARAMS["OP_H"] / MAIN_PARAMS["dt"])):
            if reactor.nA_dos >= OP_PARAMS["nA_feed"]:
                OP = 0

            if reactor.nA_dos >= OP_PARAMS["nA_feed"]*0.95 or \
                    reactor.n_B <= reactor.nB_0*0.30: #20% konverziónál szelepváltás
                br = 1

            #REACTOR STEP
            reactor.step(OP, OP_c, OP_PARAMS)

            if open_loop_on == 1 or Master_on == 0:
                reactor.TR = 363

            if show:
                # Record data
                traj[0].append(reactor.time)
                traj[1].append(SP)
                traj[2].append(reactor.TR)
                traj[3].append(OP)
                traj[5].append(reactor.n_A)
                traj[6].append(reactor.n_B)
                traj[7].append(OP_c)
                if reactor.op_mode == 0:
                    traj[8].append(float(action))
                    traj[10].append(float(reactor.TJ))
                else:
                    traj[9].append(float(action))
                    traj[10].append(float(SP_slave))

                    PID_traj[0].append(reactor.time)
                    PID_traj[1].append(SP)
                    PID_traj[2].append(PV)
                    PID_traj[3].append(SP_slave)
                    PID_traj[4].append(reactor.TJ)
                    PID_traj[5].append(OP_c)


                traj[4].append(reactor.TJ)


    #Calculate rewards
        if int(reactor.time) % rl_sampl == 0:
            PV = reactor.TR
            if not True:
                print('jhajj jhajj')
            else:
                if PV > REACTOR_PARAMS["MAT"]:
                    reward = -100
                elif np.abs(PV - SP) <= MAIN_PARAMS["EPS_PV"]:
                    reward = 100
                else:
                    reward = -np.abs(SP-PV)


            c_A = reactor.n_A / reactor.V
            c_B = reactor.n_B / reactor.V
            UA = reactor.UA
            error_master[0] = error_master[1]
            error_master[1] = (SP - PV)


            if reactor.op_mode == 0:
                new_obs = (float(PV), float(OP), float(reactor.TJ), float(c_A), float(c_B), float(UA), error_master[0], error_master[1])
                # agent_feed.remember(obs, action, reward, new_obs, 0)
                obs_feed = new_obs
                # agent_feed.learn()
            else:

                # new_obs = (float(PV), float(OP), float(SP_slave), float(reactor.TJ), float(c_A), float(c_B), float(UA), error_master[0], error_master[1])
                new_obs = (float(PV), float(OP), float(SP_slave), float(reactor.TJ), float(c_A), float(c_B), float(UA))
                # new_obs = (error_master[0], error_master[1])
                agent_cool.remember(obs_cool, action, reward, new_obs, 0)
                obs_cool = new_obs
                agent_cool.learn()




            if reactor.op_mode == 1:
                episode_reward += reward

            if reactor.n_B <= reactor.nB_0*0.10 or br == 1:
                reactor.op_mode = 1
                if cr == 0:
                    SP_slave = reactor.TJ
                cr += 1


    ep_end_time = datetime.now()
    episode_rewards.append(episode_reward)
    print('episode: ', episode, 'score %.2f' % episode_reward,
          ' Calc_Duration: ' + str(ep_end_time-ep_start_time), 'Avg score: %.2f' % np.mean(episode_rewards[-MAIN_PARAMS['SHOW_EVERY']:]))

    a1 = np.mean(episode_rewards[-MAIN_PARAMS['SHOW_EVERY']:])
    if a1 > best_ag_r and episode != 0:
        agent_cool.save_models()
        best_ag_r = a1

    if show:
        plt.figure(fignum)
        plt.clf()
        plt.suptitle('Episode: ' + str(episode) + 'AcT_reward %.2f' % episode_reward)
        # plt.suptitle("Mean reward: " + " at " + str(episode) + " Episode: " + str(
        #     np.mean(episode_rewards[-MAIN_PARAMS['SHOW_EVERY']:]))
        #              + '\n Epsilon: %.2f' % agent_feed.epsilon + '\n Epsilon_c: %.2f' % agent_cool.epsilon
        #              + '\n Reward_r: %.2f' %episode_reward, fontsize=30)
        plt.subplot(3, 2, 1)
        plt.plot(traj[0], traj[2], 'green')
        plt.plot(traj[0], traj[1], 'blue')
        plt.plot(traj[0], traj[10], 'red')

        plt.plot(traj[0], traj[4], 'black')
        plt.subplot(3,2,3)
        plt.plot(traj[0], traj[3], 'red')
        plt.plot(traj[0], traj[7], 'green')

        plt.subplot(3, 2, 2)
        plt.plot(traj[0], traj[5],'green')
        plt.plot(traj[0], traj[6], 'yellow')



        plt.subplot(3,2,5)
        plt.plot(traj[0][0:len(traj[8])], traj[8], color='red')
        plt.plot(traj[0][len(traj[8])-1:], traj[9], color='blue')


        moving_avg = np.convolve(episode_rewards, np.ones((MAIN_PARAMS["SHOW_EVERY"],)) / MAIN_PARAMS["SHOW_EVERY"],
                                 mode='valid')

        plt.subplot(3, 2, 4)
        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward {MAIN_PARAMS['SHOW_EVERY']}ma")
        plt.xlabel("episode #")
        # plt.ylim([-2000, 3000])

        plt.pause(2)
        plt.show()


        # #
        # fig, ax = plt.subplots(2, sharey=True)
        #
        #
        # # plt.subplot(2, 1, 1)
        #
        # ax[0].plot(PID_traj[0], PID_traj[1], 'green')
        # ax[0].plot(PID_traj[0], PID_traj[2], 'blue')
        # ax[0].plot(PID_traj[0], PID_traj[3], 'red')
        # ax[0].legend(['SP', 'PV', 'OP'])
        #
        # ax1=ax[1].twinx()
        # ax[1].plot(PID_traj[0], PID_traj[3], 'green')
        # ax[1].plot(PID_traj[0], PID_traj[4], 'blue')
        # ax1.plot(PID_traj[0], PID_traj[5], 'red')
        # ax[1].legend(['SP', 'PV'])
        # ax1.legend('OutPut')


end_time = datetime.now()
print('Eddig tartott a tanítás: ' + str(end_time - start_time))
#


with open('Files/' + agent_cool.name + '_rewards.pickle', 'wb') as f:
    pickle.dump(episode_rewards, f)
