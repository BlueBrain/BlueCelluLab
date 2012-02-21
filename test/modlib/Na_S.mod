:Comment : Modified from Na. Vshifted by -10. Added 2 to hinf slope.
:Reference :Kinetics were fit to data from Huguenard et al. (1988) and Hamill et al. (1991)

NEURON	{
	SUFFIX Na_S
	USEION na READ ena WRITE ina
	RANGE gNa_Sbar, gNa_S, ina
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNa_Sbar = 0.00001 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNa_S	(S/cm2)
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
	gNa_S = gNa_Sbar*m*m*m*h
	ina = gNa_S*(v-ena)
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
        v = v - 10
        if(v == -35){
            v = v+0.0001
        }
		mAlpha = (0.182 * (v- -35))/(1-(exp(-(v- -35)/9)))
		mBeta  = (0.124 * (-v -35))/(1-(exp(-(-v -35)/9)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
        if(v == -50){
            v = v + 0.0001
        }
		hAlpha = (0.024 * (v- -50))/(1-(exp(-(v- -50)/5)))
        if(v == -75){
            v = v+0.0001
        }
		hBeta  = (0.0091 * (-v - 75))/(1-(exp(-(-v - 75)/5)))
		hInf = 1.0/(1+exp((v- -65)/(8.2)))
		hTau = (1/(hAlpha + hBeta))
		
	v = v + 10
	UNITSON
}

