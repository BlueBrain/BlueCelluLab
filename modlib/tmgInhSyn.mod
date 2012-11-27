COMMENT

Derived from tmgsyn.mod from
http://senselab.med.yale.edu/senselab/modeldb/ShowModel.asp?model=3815
by Michael Hines. The only substantive change is replacement of
the single exponential behavior of conductance by a two time constant
conductance change analogous to Exp2Syn. i.e. difference of exponentials
where the difference approaches an alpha function when tau_r is close to
tau_d. Non-substantive changes are the renaming of several parameters:
tau_1 -> tau_d
tau_facil -> Fac
tau_rec -> Dep
U -> Use
and the factorization of gmax out of the weight (so the latter is dimensionless)
ENDCOMMENT

COMMENT
Revised 12/15/2000 in light of a personal communication 
from Misha Tsodyks that u is incremented _before_ x is 
converted to y--a point that was not clear in the paper.
If u is incremented _after_ x is converted to y, then 
the first synaptic activation after a long interval of 
silence will produce smaller and smaller postsynaptic 
effect as the length of the silent interval increases, 
eventually becoming vanishingly small.

Implementation of a model of short-term facilitation and depression 
based on the kinetics described in
  Tsodyks et al.
  Synchrony generation in recurrent networks 
  with frequency-dependent synapses
  Journal of Neuroscience 20:RC50:1-5, 2000.
Their mechanism represented synapses as current sources.
The mechanism implemented here uses a conductance change instead.

The basic scheme is

x -------> y    Instantaneous, spike triggered.
                Increment is u*x (see discussion of u below).
                x == fraction of "synaptic resources" that have 
                     "recovered" (fraction of xmtr pool that is 
                     ready for release, or fraction of postsynaptic 
                     channels that are ready to be opened, or some 
                     joint function of these two factors)
                y == fraction of "synaptic resources" that are in the 
                     "active state."  This is proportional to the 
                     number of channels that are open, or the 
                     fraction of max synaptic current that is 
                     being delivered. 
  tau_d
y -------> z    z == fraction of "synaptic resources" that are 
                     in the "inactive state"

  Dep
z -------> x

where x + y + z = 1

The active state y is multiplied by a synaptic weight to compute
the actual synaptic conductance (or current, in the original form 
of the model).

In addition, there is a "facilition" term u that 
governs the fraction of x that is converted to y 
on each synaptic activation.

  -------> u    Instantaneous, spike triggered, 
                happens _BEFORE_ x is converted to y.
                Increment is Use*(1-u) where Use and u both 
                lie in the range 0 - 1.
  Fac
u ------->      decay of facilitation

This implementation for NEURON offers the user a parameter 
u0 that has a default value of 0 but can be used to specify 
a nonzero initial value for u.

When Fac = 0, u is supposed to equal Use.

Note that the synaptic conductance in this mechanism 
has the same kinetics as y, i.e. decays with time 
constant tau_d.

This mechanism can receive multiple streams of 
synaptic input via NetCon objects.  
Each stream keeps track of its own 
weight and activation history.

The printf() statements are for testing purposes only.
ENDCOMMENT


NEURON {
	POINT_PROCESS tmgInhSyn
	RANGE e, i, gmax, g
	RANGE tau_r, tau_d, Dep, Fac, Use, u0,tot_dep, z0
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	: e = -70 mV for inhibitory synapses 
	:     0 mV for excitatory
	e = -80 (mV)
	: tau_d was the same for inhibitory and excitatory synapses
	: in the models used by T et al.
	tau_d = 8 (ms) < 1e-9, 1e9 >
	tau_r = 0.2 (ms): conductance rise time.
	: Dep = 100 ms for inhibitory synapses,
	:           800 ms for excitatory
	Dep = 100 (ms) < 1e-9, 1e9 >
	: Fac = 1000 ms for inhibitory synapses,
	:             0 ms for excitatory
	Fac = 1000 (ms) < 0, 1e9 >
	: Use = 0.04 for inhibitory synapses, 
	:     0.5 for excitatory
	: the (1) is needed for the < 0, 1 > to be effective
	:   in limiting the values of Use and u0
	Use = 0.04 (1) < 0, 1 >
	: initial value for the "facilitation variable"
	u0 = 0 (1) < 0, 1 >
	gmax = .001 (uS) : peak conductance
        : initial value for the "depression variable"
        z0 = 0
}

ASSIGNED {
	v (mV)
	i (nA)
	x
	g (uS)
	factor
    tot_dep
}

STATE {
	A (1)
	B (1)
}

INITIAL {
	LOCAL tp
	if (tau_r/tau_d > .9999) {
		tau_r = .9999*tau_d
	}
	A = 0
	B = 0
	g = gmax*(B - A)
	tp = (tau_r*tau_d)/(tau_d - tau_r) * log(tau_d/tau_r)
	factor = -exp(-tp/tau_r) + exp(-tp/tau_d)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tau_r
	B' = -B/tau_d
	g = gmax*(B - A)
}

NET_RECEIVE(weight (1), y, z, u, tsyn (ms)) {
INITIAL {
: these are in NET_RECEIVE to be per-stream
	y = 0
	z = z0
:	u = 0
	u = u0
	tsyn = t
: this header will appear once per stream
: printf("t\t t-tsyn\t y\t z\t u\t newu\t g\t dg\t newg\t newy\n")
}

	: first calculate z at event-
	:   based on prior y and z
	z = z*exp(-(t - tsyn)/Dep)
	z = z + ( y*(exp(-(t - tsyn)/tau_d) - exp(-(t - tsyn)/Dep)) / ((tau_d/Dep)-1) )
	: now calc y at event-
	y = y*exp(-(t - tsyn)/tau_d)

	x = 1-y-z

	: calc u at event--
	if (Fac > 0) {
		u = u*exp(-(t - tsyn)/Fac)
	} else {
		u = Use
	}

: printf("%g\t%g\t%g\t%g\t%g", t, t-tsyn, y, z, u)

	if (Fac > 0) {
		u = u + Use*(1-u)
	}
    tot_dep=x*u
: printf("\t%g\t%g\t%g", u, g, weight*x*u)

	A = A + weight*tot_dep*factor
	B = B + weight*tot_dep*factor
	y = y + x*u

	tsyn = t

: printf("\t%g\t%g\n", g, y)
}
