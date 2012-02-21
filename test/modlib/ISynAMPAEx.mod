NEURON {
	POINT_PROCESS SynAMPAEx
	RANGE e, i, gmax, g
	RANGE tau_r
	RANGE tau_d
	RANGE Dep, Fac, Use
	RANGE dist :distance from the soma
	RANGE ndist :normalized distance from the soma
	RANGE factor
	RANGE x,y,z,u,tsyn,mg
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	e = 0			(mV)	
	tau_r = 0.24755	 	(ms) 	
	tau_d = 4.8679 		(ms) 	
	Dep = 100 		(ms) 	
	Fac = 1000 		(ms) 	
	Use = 0.04 		(1) 	
	u0 = 0 			(1) 	
	factor
	gmax = 0.3      (uS)
	x
	y
	z
	u
	tsyn
	mg			(mM)
	mggate
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
}

STATE {
	A
	B
}

FUNCTION get_factor(tau_fast,tau_slow){
	LOCAL tp
	tp = (tau_fast*tau_slow)/(tau_slow - tau_fast) * log(tau_slow/tau_fast)
	get_factor = -exp(-tp/tau_fast) + exp(-tp/tau_slow)
	get_factor = 1/get_factor
}

INITIAL {
	LOCAL tp
	A = 0
	B = 0
	g = 0
	factor = get_factor(tau_r,tau_d)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	: from Jahr & Stevens
	mggate = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
	g = gmax*(B - A)
	i = g*(v - e)*mggate
}

DERIVATIVE state {
	A' = -A/tau_r
	B' = -B/tau_d
}


NET_RECEIVE(w) {
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
	if(Fac > 0){
		u = u + Use*(1-u)
	}
	
	y = y + x*u
	tsyn = t


	A = A + x*u*factor
	B = B + x*u*factor
}

INCLUDE "synutils.inc"


