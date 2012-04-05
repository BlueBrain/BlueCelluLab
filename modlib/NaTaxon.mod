:Reference :Kole 2008

NEURON	{
	SUFFIX NaTaxon
	USEION na READ ena WRITE ina
	RANGE gNaTaxonbar, gNaTaxon, ina
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTaxonbar = 0.00001 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTaxon	(S/cm2)
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
	gNaTaxon = gNaTaxonbar*m*m*m*h
	ina = gNaTaxon*(v-ena)
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
		mInf = 1.0/(1+exp((v- -31.1)/-6.5))
		mTau = 1/(mAlpha + mBeta)

    if(v == -66){
      v = v + 0.0001
    }
		hAlpha = (-0.015 * (v- -66))/(1-(exp((v- -66)/5.3)))
		hBeta  = (-0.015 * (-v -66))/(1-(exp((-v -66)/5.3)))
		hInf = 1.0/(1+exp((v- -58.7)/6.9))
		hTau = (1/(hAlpha + hBeta))
	UNITSON
}