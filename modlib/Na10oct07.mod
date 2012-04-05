:Comment :
:Reference :Kinetics were fit to data from Huguenard et al. (1988) and Hamill et al. (1991)

NEURON	{
	SUFFIX Na10oct07
	USEION na READ ena WRITE ina
	RANGE gNa10oct07bar, gNa10oct07, ina, vshift, mshift, hshift, mtfact, htfact, mtshift, htshift,vs, hslopechange,htaumin
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNa10oct07bar = 0.00001 (S/cm2)
	vshift = -10 (mV)
	mshift = 0 (mV)
	hshift = 0 (mV)
	mtfact = 1
	htfact = 1
	mtshift = 0
	htshift = 0
	vs = 0
	hslopechange = 0
	htaumin = 0
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNa10oct07	(S/cm2)
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
	gNa10oct07 = gNa10oct07bar*m*m*m*h
	ina = gNa10oct07*(v-ena)
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
        v = v + vshift
        v = v + vs
        if(v == -35){
            v = v+0.0001
        }
		:v = v + mshift
		mAlpha = (0.182 * (v- -35))/(1-(exp(-(v- -35)/9)))
		mBeta  = (0.124 * (-v -35))/(1-(exp(-(-v -35)/9)))
		:v = v - mshift
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = mtfact*1/(mAlpha + mBeta)
        if(v == -50){
            v = v + 0.0001
        }
        	:v = v + hshift
		hAlpha = (0.024 * (v- -50))/(1-(exp(-(v- -50)/5)))
        if(v == -75){
            v = v+0.0001
        }
		hBeta  = (0.0091 * (-v - 75))/(1-(exp(-(-v - 75)/5)))
		:ORIGINAL : hInf = 1.0/(1+exp((v- -65)/(6.2+hslopechange)))
		hInf = 1.0/(1+exp((v- -65)/(8.2+hslopechange)))
		hTau = htfact*(1/(hAlpha + hBeta))
		if (hTau < htaumin && v > -30)	{hTau=htaumin}
		:v = v - hshift
	v = v - vshift
	v = v - vs
	mTau = mTau + mtshift
	hTau = hTau + htshift
	UNITSON
}
