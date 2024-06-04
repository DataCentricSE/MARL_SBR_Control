from params import REACTOR_PARAMS, INIT_PARAMS, REACTION_PARAMS, MAIN_PARAMS, OP_PARAMS
import numpy as np

class Reactor:
    def __init__(self,REACTOR_PARAMS, INIT_PARAMS):

        #Init variables
        self.UA_0 = INIT_PARAMS["UA_0"]
        self.V_0 = INIT_PARAMS["V_0"]
        self.nA_0 = INIT_PARAMS["nA_0"]
        self.nB_0 = INIT_PARAMS["nB_0"]
        self.nC_0 = INIT_PARAMS["nC_0"]
        self.TR_0 = INIT_PARAMS["TR_0"]
        self.TJ_0 = INIT_PARAMS["TJ_0"]
        self.nA_dos = 0

        #Init parameters
        self.Vj = REACTOR_PARAMS["Vj"]
        self.rho_cp = REACTOR_PARAMS["rho_cp"]
        self.rho_cp_j = REACTOR_PARAMS["rho_cp_j"]
        self.MAT = REACTOR_PARAMS["MAT"]

        #State variables
        self.V = self.V_0
        self.n_A = self.nA_0
        self.n_B = self.nB_0
        self.n_C = self.nC_0
        self.TR = self.TR_0
        self.TJ = self.TJ_0
        self.UA = self.UA_0

        self.op_mode = 0

        self.time = 0

    def step(self, OP, OP_c, OP_PARAMS): #Discretized different equations are here, expl. Euler
        k0, EA_R, dHr = self.reaction_params(REACTION_PARAMS)
        V, n_A, n_B, n_C, TR, TJ, UA = self.get_vars()
        Fmax, cA_feed, Fj, T_in, Tj_in = self.op_vars(OP_PARAMS)
        Fmax = Fmax/3600     #m3/s


        if self.op_mode == 0:
            Fj = Fj/3600   #m3/s
        else:
            Fj = OP_c / 100 * Fj / 3600  # m3/s

        F = OP / 100 * Fmax

        dt = MAIN_PARAMS["dt"]

        c_A = n_A / V
        c_B = n_B / V

        r = k0 * np.exp(-EA_R / TR) * c_A * c_B
        UA = self.UA_0 * V / self.V_0
        self.UA = UA

        #Differentiate equations - Explicit euler solution
        V += dt * F
        n_A += dt * (F * cA_feed - V * r)
        n_B += - dt * V * r
        n_C += dt * V * r
        TR += dt * ((1/V/self.rho_cp) * (F * self.rho_cp * (T_in-TR) + (-dHr) * V * r - UA *(TR - TJ) ))
        TJ += dt * ((1/self.Vj/self.rho_cp_j) * (Fj * self.rho_cp_j * (Tj_in-TJ) + UA *(TR - TJ) ))

        self.nA_dos += dt * F * cA_feed
        self.time += dt
        self.update_variables(V, n_A, n_B, n_C, TR, TJ)


    def reaction_params(self, REACTION_PARAMS):
        return REACTION_PARAMS["k0"], REACTION_PARAMS["EA_R"], REACTION_PARAMS["dHr"]

    def get_vars(self):
        return self.V, self.n_A, self.n_B, self.n_C, self.TR, self.TJ, self.UA

    def op_vars(self, OP_PARAMS):
        return OP_PARAMS["Fmax"], OP_PARAMS["cA_feed"], OP_PARAMS["Fj"], OP_PARAMS["T_in"], OP_PARAMS["Tj_in"]

    def update_variables(self, V, n_A, n_B, n_C, TR, TJ):
        self.V = V
        self.n_A = n_A
        self.n_B = n_B
        self.n_C = n_C
        self.TR = TR
        self.TJ = TJ

    # def diff_eqs(self,y,t):
    #     V, n_A, n_B, n_C, TR, TJ = y
    #
    #     # dydt = [F, F * cA_feed - V * r, V * r, V * r,
    #     #         1/V/self.rho_cp) * (F * self.rho_cp * (T_in-TR) + (-dHr) * V * r - UA *(TR - TJ) ),
    #     #         (1/self.Vj/self.rho_cp_j) * (Fj * self.rho_cp_j * (Tj_in-TJ) + UA *(TR - TJ) )]

    def reset(self): #Modell inicializálása
        self.UA = INIT_PARAMS["UA_0"]
        self.V = INIT_PARAMS["V_0"]
        self.n_A = INIT_PARAMS["nA_0"]
        self.n_B = INIT_PARAMS["nB_0"]
        self.n_C = INIT_PARAMS["nC_0"]
        self.TR = INIT_PARAMS["TR_0"]
        self.TJ = INIT_PARAMS["TJ_0"]
        self.nA_dos = 0
        self.op_mode = 0

        self.time = 0


