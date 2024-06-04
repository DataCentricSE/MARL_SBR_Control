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
from Agent_TDDPG import Agent as Agent_tddpg
import numpy as np
from datetime import datetime
from PID_controller import PID
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

MAIN_PARAMS["SHOW_EVERY"] = 50

episodes = 1
observation_space_dim_feed = 8

agent_feed = Agent(alpha=0.0001, beta=0.001, input_dims=[observation_space_dim_feed],
                  tau=0.001, batch_size=64, fc1_dims=128, fc2_dims=128,
                  n_actions=1, name='2_128')

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

    # reactor.TR = np.random.randint(333,364)
    # reactor.n_B = np.random.randint(4,5)
    reactor.TR = 340
    reactor.n_B = 4.5
    OP_PARAMS["nA_feed"] = reactor.n_B

    PV = reactor.TR
    OP = 0
    c_A = reactor.n_A/reactor.V
    c_B = reactor.n_B/reactor.V
    OP_c = 100
    UA = reactor.UA

    traj = [[reactor.time], [SP], [PV], [OP], [reactor.TJ], [reactor.n_A], [reactor.n_B], [OP_c], [0], [0]]

    PID_traj = [[reactor.time], [SP], [PV], [320], [reactor.TJ], [OP_c]]
    obs_feed = (PV, OP, reactor.TJ, c_A, c_B, UA, error_master[0], error_master[1])

    if episode % MAIN_PARAMS["SHOW_EVERY"] == 0:
        print(f"{MAIN_PARAMS['SHOW_EVERY']} ep mean: {np.mean(episode_rewards[-MAIN_PARAMS['SHOW_EVERY']:])}")
        show = True
    else:
        show = False



    for i in range(int(4*3600/MAIN_PARAMS["OP_H"])):
        if reactor.op_mode == 1 and cr == 1:
            PID_traj[0][0] = reactor.time
            PID_traj[2][0] = PV
            PID_traj[3][0] = SP_slave
            PID_traj[4][0] = reactor.TJ
            PID_traj[5][0] = OP_c

        if reactor.op_mode == 0:    #RL - FEED VALVE control
            if int(reactor.time) % 100 == 0:
                action = agent_feed.choose_action(obs_feed)
                OP_b = OP
                OP += float(action)
                OP = 0 if OP < 0 else OP
                OP = 100 if OP > 100 else OP
                OP_c = 100
        else:                      #PID - COOLING AGENT VALVE control
            error_master[0] = error_master[1]
            error_master[1] = (SP - PV)
            if PID_mode == 0:
                error_master = list(np.array(error_master)*-1)
                action = PID_master.update(error_master[-2:])
            else:
                if open_loop_on == 0:
                    if Master_on == 1:
                        if open_loop_on_master == 0:
                            SP_slave_d = PID_master.update(error_master[-2:])
                            SP_slave += SP_slave_d
                            SP_slave = 370 if SP_slave>370 else SP_slave
                            SP_slave = 298 if SP_slave < 298 else SP_slave
                        else:
                            if reactor.time <14400:
                                SP_slave = 363
                            else:
                                if t_openloop % 5000 == 0 and cr != 1:
                                    SP_slave -= 10
                                SP_slave = 300 if SP_slave<300 else SP_slave
                                t_openloop += MAIN_PARAMS["OP_H"]

                    else:
                        SP_slave = 340
                    # SP_slave = 330
                    error_slave[0] = error_slave[1]
                    error_slave[1] = -(SP_slave - reactor.TJ)
                    action = PID_slave.update(error_slave[-2:])

                    action = d_lim if action>d_lim else action
                    action = -d_lim if action<-d_lim else action
                    OP_c += float(action)
                    OP_c = 0 if OP_c<0 else OP_c
                    OP_c = 100 if OP_c>100 else OP_c
                else:
                    # reactor.TR = 363
                    if t_openloop % 1200 == 0 and cr != 1:
                        OP_c -= 20
                    OP_c = 0 if OP_c<0 else OP_c
                    t_openloop += MAIN_PARAMS["OP_H"]





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
                else:
                    traj[9].append(float(action))
            if True:
                traj[4].append(reactor.TJ)

            if reactor.op_mode == 1:
                PID_traj[0].append(reactor.time)
                PID_traj[1].append(SP)
                PID_traj[2].append(PV)
                PID_traj[3].append(SP_slave)
                PID_traj[4].append(reactor.TJ)
                PID_traj[5].append(OP_c)




    #Calculate rewards
        PV = reactor.TR
        if not True:
            print('jhajj jhajj')
        else:
            if PV > REACTOR_PARAMS["MAT"]:
                reward = -np.square(SP - PV)
            elif np.abs(PV - SP) <= MAIN_PARAMS["EPS_PV"]:
                reward = 100
            else:
                reward = -np.square(SP-PV)


        c_A = reactor.n_A / reactor.V
        c_B = reactor.n_B / reactor.V
        UA = reactor.UA


        if reactor.op_mode == 0:
            if int(reactor.time) % 100 == 0:
                error_master[0] = error_master[1]
                error_master[1] = (SP - PV)
                new_obs = (float(PV), float(OP), float(reactor.TJ), float(c_A), float(c_B), float(UA), error_master[0], error_master[1])
                # agent_feed.remember(obs, action, reward, new_obs, 0)
                obs_feed = new_obs
                # agent_feed.learn()
        else:
            pass


        if reactor.op_mode == 1:
            if int(reactor.time) % 100 == 0:
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

    if show:
        plt.figure(100)
        plt.clf()
        plt.suptitle('Episode: ' + str(episode) + 'AcT_reward %.2f' % episode_reward)
        # plt.suptitle("Mean reward: " + " at " + str(episode) + " Episode: " + str(
        #     np.mean(episode_rewards[-MAIN_PARAMS['SHOW_EVERY']:]))
        #              + '\n Epsilon: %.2f' % agent_feed.epsilon + '\n Epsilon_c: %.2f' % agent_cool.epsilon
        #              + '\n Reward_r: %.2f' %episode_reward, fontsize=30)
        plt.subplot(3, 2, 1)
        plt.plot(traj[0], traj[2], 'green')
        plt.plot(traj[0], traj[1], 'blue')

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
        plt.ylim([-2000, 3000])

        plt.pause(2)
        plt.show()

        PID_traj[0] = list(np.array(PID_traj[0]) / 3600)

        # plt.figure(102)
        fig, ax = plt.subplots(2, sharey=False)
        ax2 = ax[0].twinx()

        ax[0].plot(PID_traj[0], np.array(PID_traj[1])-0.5, 'b--')
        lns1 = ax[0].plot(PID_traj[0], np.array(PID_traj[1]), 'blue', label='$SP_{master}$')
        ax[0].plot(PID_traj[0], np.array(PID_traj[1]) + 0.5, 'b--')
        lns2 = ax[0].plot(PID_traj[0], PID_traj[2], 'green', label='$PV_{master}$')
        lns3 = ax2.plot(PID_traj[0], PID_traj[3], 'red', label='$OP_{master}$')
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax[0].set_ylim([360, 365])
        ax[0].legend(lns, labs, loc=1)


        # ax[0].legend(['$SP_{master}$', '$PV_{master}$', '$OP_{master}$'])
        # ax[0].set_xlabel('Time [hr]')
        ax[0].set_ylabel('Temperature [K]')
        ax[0].set_title('Master loop')
        ax2.set_ylabel('Temperature [K]')


        ax1=ax[1].twinx()
        lns1 = ax[1].plot(PID_traj[0], PID_traj[3], 'blue', label='$SP_{slave}$')
        lns2 = ax[1].plot(PID_traj[0], PID_traj[4], 'green', label='$PV_{slave}$')
        ax[1].set_xlabel('Time [hr]')
        ax[1].set_ylabel('Temperature [K]')
        lns3 = ax1.plot(PID_traj[0], PID_traj[5], 'red', label='$OP_{slave}$')
        ax1.set_ylabel('OP [%]')

        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs, loc=1)
        ax[1].set_title('Slave loop')



end_time = datetime.now()
print('Eddig tartott a tanítás: ' + str(end_time - start_time))
#
# ##Transfer function:
# t = np.array(PID_traj[0])
# t = t-t[0]
#
# OP_c = np.array(PID_traj[5])
# TJ = np.array(PID_traj[4])
#
#
# fig, ax = plt.subplots(1, sharey=True)
# ax.plot(t, TJ)
# ax1 = ax.twinx()
# ax1.plot(t, OP_c, 'red')
#
#
# ## Másik módszer
# # specify number of steps
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# from scipy.optimize import minimize
# from scipy.interpolate import interp1d
#
# a = 20000
# b = 100000
# t = np.array(PID_traj[0])
# t = t-t[0]
# t = t[a:b:10]
# # t=t/3600
# u = OP_c[a:b:10]
# yp = TJ[a:b:10]
# u0 = u[0]
# yp0 = yp[0]
#
# ns = len(t)
# delta_t = t[1]-t[0]
# # create linear interpolation of the u data versus time
# uf = interp1d(t,u)
# x = np.array([-0.24, 184, 0])
# x2 = np.array([-0.27, 184, 0])
#
#
# # define first-order plus dead-time approximation
# def fopdt(y,t,uf,Km,taum,thetam):
#     # arguments
#     #  y      = output
#     #  t      = time
#     #  uf     = input linear function (for time shift)
#     #  Km     = model gain
#     #  taum   = model time constant
#     #  thetam = model time constant
#     # time-shift u
#     try:
#         if (t-thetam) <= 0:
#             um = uf(0.0)
#         else:
#             um = uf(t-thetam)
#     except:
#         #print('Error with time extrapolation: ' + str(t))
#         um = u0
#     # calculate derivative
#     dydt = (-(y-yp0) + Km * (um-u0))/taum
#     return dydt
#
# # simulate FOPDT model with x=[Km,taum,thetam]
# def sim_model(x):
#     # input arguments
#     Km = x[0]
#     taum = x[1]
#     thetam = x[2]
#     # storage for model values
#     ym = np.zeros(ns)  # model
#     # initial condition
#     ym[0] = yp0
#     # loop through time steps
#     for i in range(0,ns-1):
#         ts = [t[i],t[i+1]]
#         y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
#         ym[i+1] = y1[-1]
#     return ym
#
# # define objective
# def objective(x):
#     # simulate model
#     ym = sim_model(x)
#     # calculate objective
#     obj = 0.0
#     for i in range(len(ym)):
#         obj = obj + (ym[i]-yp[i])**2
#     # return result
#     print(obj)
#     return obj
#
#
#
#
# print('Kp: ' + str(x[0]))
# print('taup: ' + str(x[1]))
# print('thetap: ' + str(x[2]))
#
# # calculate model with updated parameters
# ym2 = sim_model(x)
# ym3 = sim_model(x2)
# # plt.subplot(2,1,1)
# t=t/3600
#
# fig, ax = plt.subplots(1, sharey=True)
# lns1 = ax.plot(t,yp,'k-',linewidth=2,label='Process Data')
# lns2 = ax.plot(t,ym2,'r--',linewidth=2,label='FOPDT_1')
# lns3 = ax.plot(t,ym3,'g--',linewidth=2,label='FOPDT_2')
# plt.ylabel('Temperature [K]')
# plt.xlabel('Time [hr]')
#
# ax1 = ax.twinx()
# lns4 = ax1.plot(t,uf(t*3600),'b--',linewidth=3, label='Input data')
# plt.ylabel('Output [%]')
#
# lns = lns1+lns2+lns3+lns4
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=1)
#
#
# plt.show()
#
#
# fig, ax = plt.subplots(1, sharey=True)
#
# t0 = 120000
#
# lns1 = ax.plot(PID_traj[0][t0:], PID_traj[3][t0:], 'green', label='$SP_{slave}$')
# lns2 = ax.plot(PID_traj[0][t0:], PID_traj[4][t0:], 'blue', label='PV')
# ax1 = ax.twinx()
# lns3 = ax1.plot(PID_traj[0][t0:], PID_traj[5][t0:], 'red', label='OP')
#
# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)

# plt.legend(['$SP_{slave}$','fasz','asz2'])
# ax.legend(['1','2'])

#
# ##Transfer function MASTER:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# from scipy.optimize import minimize
# from scipy.interpolate import interp1d
#
# t = np.array(PID_traj[0])
# t = t-t[0]
#
# OP = np.array(PID_traj[3])
# PV = np.array(PID_traj[2])
#
#
# plt.figure(13)
# plt.plot(t, PV)
# plt.plot(t, OP, 'red')
#
#
# a = 82000
# b = 280000
# t = np.array(PID_traj[0])
# t = t-t[0]
# t = t[a:b:10]
# u = OP[a:b:10]
# yp = PV[a:b:10]
# u0 = u[0]
# yp0 = yp[0]
#
# ns = len(t)
# delta_t = t[1]-t[0]
# # create linear interpolation of the u data versus time
# uf = interp1d(t,u)
#
# # define first-order plus dead-time approximation
# def fopdt(y,t,uf,Km,taum,thetam):
#     # arguments
#     #  y      = output
#     #  t      = time
#     #  uf     = input linear function (for time shift)
#     #  Km     = model gain
#     #  taum   = model time constant
#     #  thetam = model time constant
#     # time-shift u
#     try:
#         if (t-thetam) <= 0:
#             um = uf(0.0)
#         else:
#             um = uf(t-thetam)
#     except:
#         #print('Error with time extrapolation: ' + str(t))
#         um = u0
#     # calculate derivative
#     dydt = (-(y-yp0) + Km * (um-u0))/taum
#     return dydt
#
# # simulate FOPDT model with x=[Km,taum,thetam]
# def sim_model(x):
#     # input arguments
#     Km = x[0]
#     taum = x[1]
#     thetam = x[2]
#     # storage for model values
#     ym = np.zeros(ns)  # model
#     # initial condition
#     ym[0] = yp0
#     # loop through time steps
#     for i in range(0,ns-1):
#         ts = [t[i],t[i+1]]
#         y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
#         ym[i+1] = y1[-1]
#     return ym
#
# x = np.array([1, 1230, 0])
# # show final objective
# print('Kp: ' + str(x[0]))
# print('taup: ' + str(x[1]))
# print('thetap: ' + str(x[2]))
#
# # calculate model with updated parameters
# ym2 = sim_model(x)
# # plot results
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
# plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
# plt.ylabel('Output')
# plt.legend(loc='best')
# plt.subplot(2,1,2)
# plt.plot(t,u,'bx-',linewidth=2)
# plt.plot(t,uf(t),'r--',linewidth=3)
# plt.legend(['Measured','Interpolated'],loc='best')
# plt.ylabel('Input Data')
# plt.show()