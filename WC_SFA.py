from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models
import numpy 
from tvb.simulator.models.base import ModelNumbaDfun
from numba import guvectorize, float64

@guvectorize([(float64[:],) * 23], '(n),(m),(o)' + ',()'*19 + '->(n)', nopython=True)
def _numba_dfun(y, c, lc, c_excexc, c_inhexc, exc_ext, b, c_excinh, c_inhinh, inh_ext, 
                k_e, k_i, h_e, h_i, n, a_a, mu_a, tau_exc, tau_inh, tau_a, tau_uE, tau_uI, ydot):
    
    E_input = (c_excexc[0] * y[0] - c_inhexc[0] * y[1] + c[0] + lc[0] + lc[1] + exc_ext[0] - y[2] + y[3])
    I_input = (c_excinh[0] * y[0] - c_inhinh[0] * y[1] + lc[0] + lc[1] + inh_ext[0] + y[4])

    sigm_E = k_e[0] * numpy.power(numpy.maximum(0, E_input-h_e[0]), n[0])
    sigm_I = k_i[0] * numpy.power(numpy.maximum(0, I_input-h_i[0]), n[0])
    sigm_a = b[0] / (1.0 + numpy.exp(-a_a[0] * (y[0] - mu_a[0])))

    # Differential equations
    ydot[0] = (-y[0] + sigm_E) / tau_exc[0]
    ydot[1] = (-y[1] + sigm_I) / tau_inh[0]
    ydot[2] = (-y[2] + sigm_a) / tau_a[0]
    ydot[3] = -y[3] / tau_uE[0]
    ydot[4] = -y[4] / tau_uI[0]


class WilsonCowan_SFA(ModelNumbaDfun):
    '''
    '''
    tau_exc = NArray(
        label=":math:`\\tau_{exc}`",
        default=numpy.array([5]),
        domain=Range(lo=1.0, hi=10.0, step=0.1),
        doc="Time constant of the excitatory population [ms].")
    tau_inh = NArray(
        label=":math:`\\tau_{inh}`",
        default=numpy.array([5]),
        domain=Range(lo=1.0, hi=10.0, step=0.1),
        doc="Time constant of the inhibitory population [ms].")
    tau_a = NArray(
        label=":math:`\\tau_{inh}`",
        default=numpy.array([200.]),
        domain=Range(lo=1.0, hi=10.0, step=0.1),
        doc="Time constant of the adaptation dynamics [ms].")
    tau_uE = NArray(
        label=":math:`\\tau_{inh}`",
        default=numpy.array([5.]),
        domain=Range(lo=0, hi=50.0, step=0.1),
        doc="Noise correlation time [ms] of the exc population")
    tau_uI = NArray(
        label=":math:`\\tau_{inh}`",
        default=numpy.array([5.]),
        domain=Range(lo=0, hi=50.0, step=0.1),
        doc="Noise correlation time [ms] of the inh population")
    c_excexc = NArray(
        label=":math:`c_{ee}`",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=30.0, step=0.5),
        doc="Coupling from excitatory to excitatory population.")
    c_excinh = NArray(
        label=":math:`c_{ei}`",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=30.0, step=0.5),
        doc="Coupling from excitatory to inhibitory population.")
    c_inhexc = NArray(
        label=":math:`c_{ie}`",
        default=numpy.array([3.0]),
        domain=Range(lo=0.0, hi=30.0, step=0.5),
        doc="Coupling from inhibitory to excitatory population.")
    c_inhinh = NArray(
        label=":math:`c_{ii}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=30.0, step=0.5),
        doc="Coupling from inhibitory to inhibitory population.")
    k_e = NArray(
        label=":math:`k_{e}`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.1, hi=5.0, step=0.1),
        doc="")
    k_i = NArray(
        label=":math:`k_{i}`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.1, hi=5.0, step=0.1),
        doc="")
    h_e = NArray(
        label=":math:`h_{e}`",
        default=numpy.array([0.]),
        domain=Range(lo=0.1, hi=5.0, step=0.1),
        doc="")
    h_i = NArray(
        label=":math:`h_{i}`",
        default=numpy.array([12.]),
        domain=Range(lo=0.1, hi=5.0, step=0.1),
        doc="")
    n = NArray(
        label=":math:`n`",
        default=numpy.array([2]),
        domain=Range(lo=0.1, hi=5.0, step=0.1),
        doc="")
    a_a = NArray(
        label=":math:`a_{i}`",
        default=numpy.array([5.]),
        domain=Range(lo=0.1, hi=25.0, step=0.1),
        doc="Gain of the adaptation dynamics.")
    mu_a = NArray(
        label=":math:`\\mu_{i}`",
        default=numpy.array([1.5]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc="Firing threshold of the adaptation.")
    b = NArray(
        label=":math:`\\b`",
        default=numpy.array([0.]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="Adaptation strength.")
    exc_ext = NArray(
        label=":math:`I_{ext}^{e}`",
        default=numpy.array([3.6]),
        domain=Range(lo=-5.0, hi=5.0, step=0.1),
        doc="External input to excitatory population.")
    inh_ext = NArray(
        label=":math:`I_{ext}^{i}`",
        default=numpy.array([3.6]),
        domain=Range(lo=-5.0, hi=5.0, step=0.1),
        doc="External input to inhibitory population.")
    
    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.1, 1.]),
                 "I": numpy.array([0.1, 1.]),
                 "a": numpy.array([0.1, 1.]),
                 "uE": numpy.array([0., 0.]),
                 "uI": numpy.array([0., 0.])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "a", "uE", "uI"),
        default=("E", "I", "a", "uE", "uI"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored.""")
    state_variables = tuple('E I a uE uI'.split())
    _nvar = 5
    cvar = numpy.array([0], dtype=numpy.int32)
    stvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        E, I, a, uE, uI = state_variables
        derivative = numpy.empty_like(state_variables)
        
        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_0 = local_coupling * E
        lc_1 = local_coupling * I
                
        E_input = (self.c_excexc * E - self.c_inhexc * I + c_0 + lc_0 + lc_1 + self.exc_ext - a + uE)
        I_input = (self.c_excinh * E - self.c_inhinh * I + lc_0 + lc_1 + self.inh_ext + uI)
        
        sigm_E = self.k_e * numpy.power(numpy.maximum(0, E_input-self.h_e), self.n)
        sigm_I = self.k_i * numpy.power(numpy.maximum(0, I_input-self.h_i), self.n)
        sigm_a = self.b / (1.0 + numpy.exp(-self.a_a * (E - self.mu_a)))
        
        # Differential equations
        derivative[0] = (-E + sigm_E) / self.tau_exc
        derivative[1] = (-I + sigm_I) / self.tau_inh
        derivative[2] = (-a + sigm_a) / self.tau_a
        derivative[3] = -uE / self.tau_uE # eq. for OU noise process in the exc population (put additive noise in this state variable)
        derivative[4] = -uI / self.tau_uI # eq. for OU noise process in the inh population (put additive noise in this state variable)
                        
        return derivative
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        """
        x_ = state_variables.reshape(state_variables.shape[:-1]).T
        c_ = coupling.reshape(coupling.shape[:-1]).T
        local_coupling = numpy.array([local_coupling * state_variables[0, :], local_coupling * state_variables[1, :]])
        local_coupling_ = local_coupling.reshape(local_coupling.shape[:-1]).T
        deriv = _numba_dfun(x_, c_, local_coupling_,
                            self.c_excexc, self.c_inhexc, self.exc_ext, self.b, self.c_excinh, 
                            self.c_inhinh, self.inh_ext, self.k_e, self.k_i, self.h_e, self.h_i, 
                            self.n, self.a_a, self.mu_a, self.tau_exc, self.tau_inh, self.tau_a, self.tau_uE, self.tau_uI)
        return deriv.T[..., numpy.newaxis]

  
