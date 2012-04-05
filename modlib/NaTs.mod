:Reference :Colbert and Pan 2002

NEURON	{
	SUFFIX NaTs
	USEION na READ ena WRITE ina
	RANGE gNaTsbar, gNaTs, ina
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTsbar = 0.00001 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTs	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNaTs = gNaTsbar*m*m*m*h
	ina = gNaTs*(v-ena)
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
    if(v == -28){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -28))/(1-(exp(-(v- -28)/6.8)))
		mBeta  = (0.124 * (-v -28))/(1-(exp(-(-v -28)/6.8)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)

    if(v == -66){
      v = v + 0.0001
    }
		hAlpha = (-0.015 * (v- -66))/(1-(exp((v- -66)/5.3)))
		hBeta  = (-0.015 * (-v -66))/(1-(exp((-v -66)/5.3)))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = (1/(hAlpha + hBeta))
	UNITSON
}