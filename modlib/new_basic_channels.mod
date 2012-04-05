: Set of basic channels - Fast sodium, Pottasim delayed rectifier and Connor-Stevens A channel

NEURON	{
	SUFFIX basic
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik	: Potassium delayed rectifier part
	NONSPECIFIC_CURRENT ia	: Achannel
	RANGE gnabar, gna, ina
	RANGE gkbar,gk,ik 
	RANGE gabar,ga, ea, ia		: Achannel
	RANGE mtau_tempfactor, ntau_tempfactor, htau_tempfactor, atau_tempfactor, btau_tempfactor
    RANGE MSHFT, HSHFT, NSHFT
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gnabar = 0.01 (S/cm2)
	MSHFT  = 0.3
   	HSHFT  = -8
   	gkbar  = 0.01 (S/cm2)
	NSHFT  = -4.3
	gabar  = 0.01 (S/cm)	
	ea     = -75 (mV)		
	mtau_tempfactor = 1
	ntau_tempfactor = 1
	htau_tempfactor = 1
	atau_tempfactor = 1
	btau_tempfactor = 1
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gna	(S/cm2)
	: Potassium delayed rectifier part
	ek	(mV)
	ik	(mA/cm2)
	gk	(S/cm2)
	: A current part
	ia	(mA/cm2)
	ga	(S/cm2)
	malpha   (/ms)
	mbeta    (/ms)
	halpha   (/ms)
	hbeta    (/ms)
	minf
	mtau
	hinf
	htau
	nalpha   (/ms)
	nbeta    (/ms)
	ninf
	ntau
	ainf
	atau
	binf
	btau
}

STATE	{
	m
	h
	n
	a
	b
	}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gna = gnabar*m*m*m*h
	ina = gna*(v-ena)
	: Potassium delayed rectifier part
	gk = gkbar*(n^4)
	ik = gk*(v-ek)
	: A current part
	ga = gabar*a*a*a*b
	ia = ga*(v-ea)
}

DERIVATIVE states	{
	settables(v)
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
	: Potassium delayed rectifier part
	n' = (ninf-n)/ntau
	: A current part
	a' = (ainf - a)/atau
	b' = (binf - b)/btau
}

INITIAL	{
	settables(v)
	m = minf
	h = hinf
	n = ninf
	a = ainf
	b = binf
}

FUNCTION SafeExp(x, y, Vm) {
   if(fabs(Vm) > 1e-6) {
        SafeExp = (x * Vm)/(exp(Vm/y) - 1)
   } else {
        SafeExp = x/((Vm/y/2) - 1)
   }
}

PROCEDURE settables(v (mV)) {
   : TABLE malpha, mbeta, halpha, hbeta, minf, mtau, hinf, htau, nalpha, nbeta, ninf, ntau, ainf, atau, binf, btau  
    :      FROM -100 TO 100 WITH 200
    malpha = SafeExp(0.1,10,-(v+35+MSHFT))
    mbeta = 4*exp(-(v+60+MSHFT)/18)
    halpha = 0.07*exp(-(v+60+HSHFT)/10)
    hbeta = 1/(exp(-((v+30+HSHFT)/10)+1))
    minf = malpha/(malpha + mbeta)
    mtau = mtau_tempfactor*(1/3.8)*1/(malpha + mbeta)
    hinf = halpha/(halpha + hbeta)
    htau = htau_tempfactor*(1/3.8)*1/(halpha + hbeta)
    nalpha = SafeExp(0.01,10,-(v+50+NSHFT))
    nbeta = 0.125*exp(-(v+60+NSHFT)/80)
    ninf = nalpha/(nalpha + nbeta)
    ntau = ntau_tempfactor*(2/3.8)*1/(nalpha + nbeta)
    ainf = (0.0761*(exp((v+94.22)/31.84))/(1+exp((v+1.17)/28.93)))^(1/3)
    atau = atau_tempfactor*(0.3632+(1.158/(1+exp((v+55.96)/20.12))))
    binf =  (1/(1+exp((v+53.3)/14.54))^4)
    btau = btau_tempfactor*(1.24 + 2.678/(1+exp((v+50)/16.027)))
}
