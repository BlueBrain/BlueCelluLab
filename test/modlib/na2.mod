:Comment :
:Reference :Traub et. al. J. Neurophysiol, 2003, 89: 909-921

NEURON	{
	SUFFIX na2
	USEION na READ ena WRITE ina
	RANGE gna2bar, gna2, ina 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gna2bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gna2	(S/cm2)
	mInf
	mTau
	hInf
	hTau
}

STATE	{ 
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gna2 = gna2bar*m*m*m*h
	ina = gna2*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
	UNITSOFF
		mInf = 1/(1+exp((-v-34.5)/10))
        if(v<-26.5){
                mTau = 0.025+0.14*exp((v+26.5)/10)
        }else{
                mTau = 0.02+0.145*exp((-v-26.5)/10)
        }
		hInf = 1/(1+exp((v+59.4)/10.7))
		hTau = 0.15+1.15/(1+exp((v+33.5)/15))
	UNITSON
}
