NEURON {
	POINT_PROCESS SynNMDAEx
	RANGE e, i, gmax, g
	RANGE tau_r
	RANGE tau_d
	RANGE Dep, Fac, Use, Inact
	RANGE dist :distance from the soma
	RANGE ndist :normalized distance from the soma
	RANGE factor
	RANGE x,y,z,u,tsyn,mg
	RANGE random_interval
	
	RANGE lSF,gSF,Vtrg	:local and global synaptic homeostatic plasticity factor between 0-1
	RANGE tau_H	: time constant for the homeostasis, should be orders of magnitude larger than the membrane time constant.

	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	e = 0			(mV)	
	tau_r = 0.29	(ms) 	
	tau_d = 43		(ms) 
	:rise and decay time constants obtained from Sarid et al '07
	Inact = 3		(ms)
	Dep = 800 		(ms) 	
	Fac = 3 		(ms) 	
	
	Use = 0.67 		(1) 	
	factor
	gmax = 0.001    (uS)
	x
	y
	z
	u
	tsyn
	mg = 1			(mM)
	mggate
	Vtrg = -60
	:the default is that there is not homeostasis with such a high time constant
	tau_H = 1e300	 	(ms) :assuming membrane potential within the range of 20ms tau_H is three orders of magnitude higher. 	
	random_interval :if set to a >0 values will start to get random presynaptic stimulation at 1/random_interval rate

}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
}

STATE {
	A
	B
	lSF
}

FUNCTION get_factor(tau_fast,tau_slow){
	LOCAL tp
	tp = (tau_fast*tau_slow)/(tau_slow - tau_fast) * log(tau_slow/tau_fast)
	get_factor = -exp(-tp/tau_fast) + exp(-tp/tau_slow)
	get_factor = 1/get_factor
}

INITIAL {
	LOCAL tp
	:start fresh
	x = 0
	y = 0
	z = 0
	u = 0
	A = 0
	B = 0
	g = 0
	lSF = 1
	factor = get_factor(tau_r,tau_d)
	if(random_interval>0){
		net_send(exprand(random_interval),-1)
	}
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	: from Jahr & Stevens
	mggate = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
	lSF = lSF*(lSF>0)
	g = gmax*(B - A)*mggate
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tau_r
	B' = -B/tau_d
	lSF' = (Vtrg-v)/tau_H :the global implementation requires a pointer to the somatic voltage.
}



NET_RECEIVE(weight,y,z,u,tsyn (ms)) {
    weight = weight*0.71 
    :the NETCON.weight = gsyn (per synaptic contact) * scaling factor * 0.71, as gNMDA=0.71*gAMPA from Chaelon et al.'03, and Markram et al. '97
	: first calculate z at event-
	:   based on prior y and z
    tsyn = t
	z = z*exp(-(t - tsyn)/Dep)
	z = z + ( y*(exp(-(t - tsyn)/Inact) - exp(-(t - tsyn)/Dep)) / ((Inact/Dep)-1) )
	: now calc y at event-
	y = y*exp(-(t - tsyn)/Inact)
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


	A = A + weight*x*u*factor
	B = B + weight*x*u*factor
	
	if(random_interval>0){
		net_send(exprand(random_interval),-1)
	}

	tsyn = t
	
}

INCLUDE "synutils.inc"

