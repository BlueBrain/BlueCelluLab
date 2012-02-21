: A-type channel taken from Connor and Stevens

NEURON	{
	SUFFIX csa
	NONSPECIFIC_CURRENT icsa
	RANGE gcsa, ecsa, icsa,gcsabar
	RANGE ASHFT,BSHFT
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gcsabar = 0.01 (S/cm)
	ecsa = -75 (mV)	
	ASHFT = 0
	BSHFT = 0
}

ASSIGNED	{
	v	(mV)
	icsa	(mA/cm2)
	gcsa	(S/cm2)
}

STATE	{
	a
	b
	}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gcsa = gcsabar*a*a*a*b
	icsa = gcsa*(v-ecsa)
}

DERIVATIVE states	{
    LOCAL inf, tau
    inf = ainf(v)  tau=atau(v)
    a' = (inf - a)/tau
    inf = binf(v)  tau=btau(v)
    b' = (inf - b)/tau
}

INITIAL	{
	a = ainf(v)
	b = binf(v)
}

FUNCTION gate(v){LOCAL lm,lh
	lm = ainf(v)
	lm = lm*lm*lm
	lh = binf(v)
	gate = lm*lh
}


FUNCTION ainf(Vm (mV))	{
	UNITSOFF
	ainf = (0.0761*(exp((Vm+94.22+ASHFT )/31.84))/(1+exp((Vm+1.17+ASHFT)/28.93)))^(1/3)
	UNITSON
}

FUNCTION atau(Vm (mV)) (ms)	{
	UNITSOFF
	atau = 0.3632+(1.158/(1+exp((Vm+55.96+ASHFT)/20.12)))
	UNITSON
}

FUNCTION binf(Vm (mV))	{
	UNITSOFF
	binf =  (1/(1+exp((Vm+53.3+BSHFT)/14.54))^4)
	UNITSON
}

FUNCTION btau(Vm (mV)) (ms)	{
	UNITSOFF
	btau = 1.24 + 2.678/(1+exp((Vm+50+BSHFT)/16.027))
	UNITSON
}
