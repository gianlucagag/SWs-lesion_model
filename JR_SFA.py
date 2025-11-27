from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models
import numpy

class JansenRit_SFA(models.Model):
    
    '''
        y0    pyramidal neurons post-synaptic membrane potential
        y1    excitatory interneurons post-synaptic membrane potential
        y2    inhibitory interneurons post-synaptic membrane potential
        y3    pyramidal neurons firing rate
        y4    excitatory interneurons firing rate
        y5    inhibitory interneurons firing rate
        ad    Spike-frequency adaptation (Moran et al. 2007)
    '''
    
    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")
    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")
    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")
    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")
    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].
        The usual value for this parameter is 6.0.""")
    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1].""")
    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")
    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")
    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")
    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")
    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")
    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")
    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")
    k_ad = NArray(
        label=r":math:`\k_{ad}`",
        default=numpy.array([1/512]),
        domain=Range(lo=0.0, hi=0.0005, step=0.01),
        doc="""Adaptation rate constant [ms^-1]""")
    g_ad = NArray(
        label=r":math:`\k_{ad}`",
        default=numpy.array([10]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Scale factor of adaptation (adaptation strength)""")
    
    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),
                 "ad": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("y0", "y1", "y2", "y3", "y4", "y5", "ad"),
        default=("y0", "y1", "y2", "y3"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")
    state_variables = tuple('y0 y1 y2 y3 y4 y5 ad'.split())
    _nvar = 7
    cvar = numpy.array([1, 2], dtype=numpy.int32)
    stvar = numpy.array([4], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        y0, y1, y2, y3, y4, y5, ad = state_variables
        
        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling*(y1 - y2)

        # NOTE: for local couplings
        # 0: pyramidal cells
        # 1: excitatory interneurons
        # 2: inhibitory interneurons
        # 0 -> 1,
        # 0 -> 2,
        # 1 -> 0,
        # 2 -> 0,
        
        exp = numpy.exp
        sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - 
                                                              (y1 - y2))))
        sigm_y0_1  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - 
                                                              (self.a_1 * self.J * (y0 - self.g_ad * ad) )))) #adaptation enters here
        sigm_y0_3  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - 
                                                              (self.a_3 * self.J * y0))))

        dy0 = y3
        dy1 = y4
        dy2 = y5
        dy3 = self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0
        dy4 = self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 +
                                 lrc + src) - 2.0 * self.a * y4 - self.a ** 2 * y1
        dy5 = self.B * self.b * (self.a_4 * self.J * 
                                 sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2
        dad = self.k_ad * (sigm_y0_1 - ad)

        derivative = numpy.array([dy0, dy1, dy2, dy3, dy4, dy5, dad])
        
        return derivative