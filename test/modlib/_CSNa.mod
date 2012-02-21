: Sodium channel taken from Connor and Stevens

NEURON	{
	SUFFIX csna
	USEION na READ ena WRITE ina
	RANGE gcsnabar, gna, ina
	RANGE MSHFT,HSHFT
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	MSHFT = 0.3
    HSHFT = -8
    gcsnabar = 0.01 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gcsna	(S/cm2)
}

STATE	{
	m
	h
	}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gcsna = gcsnabar*m*m*m*h
	ina = gcsna*(v-ena)
}

DERIVATIVE states	{
    LOCAL inf, tau
    inf = minf(v)  tau = mtau(v)
    m' = (inf-m)/tau
    inf = hinf(v)  tau = htau(v)
    h' = (inf-h)/tau
}

INITIAL	{
	m = minf(v)
	h = hinf(v)
}

FUNCTION gate(vv){LOCAL lm,lh
	lm = minf(vv)
	lm = lm*lm*lm
	lh = hinf(vv)
	gate = lm*lh
}

FUNCTION malpha(Vm (mV)) {
	UNITSOFF
	malpha = -0.1*(Vm+35+MSHFT)/(exp(-(Vm+35+MSHFT)/10)-1)
	UNITSON
}

FUNCTION mbeta(Vm (mV))	{
	UNITSOFF
	mbeta = 4*exp(-(Vm+60+MSHFT)/18)
	UNITSON
}

FUNCTION halpha(Vm (mV))	{
	UNITSOFF
	halpha = 0.07*exp(-(Vm+60+HSHFT)/10)
	UNITSON
}

FUNCTION hbeta(Vm (mV))	{
	UNITSOFF
	hbeta = 1/(exp(-((Vm+30+HSHFT)/10)+1))
	UNITSON
}

FUNCTION minf(Vm (mV)) {
	UNITSOFF
	minf = malpha(Vm)/(malpha(Vm)+ mbeta(Vm))
	UNITSON
}

FUNCTION mtau(Vm (mV))	{
	UNITSOFF
	mtau = (1/3.8)*1/(malpha(Vm)+ mbeta(Vm)) : where the factor is the temp factor
	UNITSON
}

FUNCTION hinf(Vm (mV)) {
	UNITSOFF
	hinf = halpha(Vm)/(halpha(Vm)+ hbeta(Vm))
	UNITSON
}

FUNCTION htau(Vm (mV))	{
	UNITSOFF
	htau = (1/3.8)*1/(halpha(Vm)+ hbeta(Vm)) : where the factor is the temp factor
	UNITSON
}
