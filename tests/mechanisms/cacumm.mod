COMMENT
	calcium accumulation into a volume of area*depth next to the
	membrane with a decay (time constant tau) to resting level
	given by the global calcium variable cai0_ca_ion
	Modified to include a resting current (irest) and peak value
	(cmax)
	i is a dummy current needed to force a BREAKPOINT
ENDCOMMENT

NEURON {
	SUFFIX cacum
	USEION ca READ ica WRITE cai
	NONSPECIFIC_CURRENT i
	RANGE depth, tau, cai0, cmax
}

UNITS {
	(mM) = (milli/liter)
	(mA) = (milliamp)
	F = (faraday) (coulombs)
}

PARAMETER {
	depth = 0.1 (um)	: assume volume = area*depth
	irest = 0  (mA/cm2)		: to be initialized in hoc	
	tau = 100 (ms)
	cai0 = 50e-6 (mM)	: Requires explicit use in INITIAL
			: block for it to take precedence over cai0_ca_ion
			: Do not forget to initialize in hoc if different
			: from this default.
}

ASSIGNED {
	ica (mA/cm2)
	cmax
	i  	 (mA/cm2)
}

STATE {
	cai (mM)
}

INITIAL {
	cai = cai0
	:irest = ica
	cmax=cai
}

BREAKPOINT {
	SOLVE integrate METHOD derivimplicit
	if (cai>cmax) {cmax=cai}
	i=0
}

DERIVATIVE integrate {
	cai' = (irest-ica)/depth/F/2 * (1e4) + (cai0 - cai)/tau
}
