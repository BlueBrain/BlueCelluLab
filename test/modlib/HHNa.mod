:Comment :
:Reference :A quantitative description of membrane current and its application   conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952)

NEURON	{
	SUFFIX HHNa
	USEION na READ ena WRITE ina
	RANGE gHHNabar, gHHNa, ina 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gHHNabar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gHHNa	(S/cm2)
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
	gHHNa = gHHNabar*m*m*m*h
	ina = gHHNa*(v-ena)
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
        if(v == 25){
            v = v + 0.0001
        }
		mAlpha = (0.1*(25-v))/(exp((25-v)/10) -1.0)
		mBeta  = 4.0 * (exp(-v/18))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
		hAlpha = 0.07 * exp(-v/20)
		hBeta  = 1/(exp((30-v)/10) + 1.0)
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = 1/(hAlpha + hBeta)
	UNITSON
}
