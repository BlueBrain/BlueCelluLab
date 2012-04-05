: Delayed Rectifier channel taken from Connors and Stevens

NEURON	{
	SUFFIX csk
	USEION k READ ek WRITE ik
	RANGE gcskbar,oinf, otau
	RANGE NSHFT,tauB, NSLOP
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gcskbar = 0.01 (S/cm2)
	NSHFT = -4.3
	NSLOP = 0
	tauB = 1 (ms)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gk	(S/cm2)
}

STATE	{ n }

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gk = gcskbar*(n^4)
	ik = gk*(v-ek)
}

DERIVATIVE states	{
    LOCAL inf, tau
    inf = ninf(v)
    tau = ntau(v)*tauB
    n' = (inf-n)/tau
}

INITIAL	{
	n = ninf(v)
}

FUNCTION gate(v){LOCAL lm,lh
	lm = ninf(v)
	gate = lm*lm*lm*lm
}

FUNCTION alpha(Vm (mV))	{
	UNITSOFF
	alpha = -0.01*(Vm+50+NSHFT)/(exp(-(Vm+50+NSHFT)/10) - 1)
	UNITSON
}

FUNCTION beta(Vm (mV)) 	{
	UNITSOFF
	beta = 0.125*exp(-(Vm+60+NSHFT)/(80+NSLOP))
	UNITSON
}

FUNCTION ninf(Vm (mV)) {
	UNITSOFF
	ninf = alpha(Vm)/(alpha(Vm)+ beta(Vm))
	UNITSON
}

FUNCTION ntau(Vm (mV))	{
	UNITSOFF
	ntau = (2/3.8)*1/(alpha(Vm)+ beta(Vm)) : where the factor is the temp factor
	UNITSON
}
