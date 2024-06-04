REACTOR_PARAMS = {
               "Vj": 0.41, # m3
               "rho_cp": 4800, #kJ/ m3C
               "rho_cp_j": 4183, #kJ/ m3C
               "MAT": 373, #Maximum Allowable Temperature, K
               }

REACTION_PARAMS = {"k0": 1.465e07,
                   "EA_R": 8500, #K
                   "dHr": -350_000, #kJ/kmol
                    }

OP_PARAMS = {"Fmax": 0.5, #feed rate
             "Fj": 4, #cooling agent flow, m3/h
             "nA_feed": 4, # feed moles of A, kmol
             "cA_feed": 5, # feed concentration, kmol/m3
             "T_in": 298, #feed temperature
             "Tj_in": 298, #cooling feed temperature
             }

INIT_PARAMS = {"V_0": 0.5, #initial volume m3
               "nA_0": 0, #initial moles of A component, kmol
               "nB_0": 5, #initial moles of B component, kmol
               "nC_0": 0, #initial moles of C component, kmol
               "TR_0": 298, #initial reactor temperature, K
               "TJ_0": 298, #initial jacket temperature, K
               "UA_0": 2, # kW/°C
               }

MAIN_PARAMS = {"EPISODES": 50_000,
               "epsilon": 0.9,
               "START_EPSILON_DECAYING": 0,
               "END_EPSILON_DECAYING": 5_00,
               "EPS_DECAY": 0.99995,
               "EPS_PV": 0.5,
               "dt": 0.1,
               "LEARNING_RATE": 0.001,
               "DISCOUNT_FACTOR": 0.99,
               "SHOW_EVERY": 50,
               "SP_c_time": 1001,
               "OP_H": 1
               }