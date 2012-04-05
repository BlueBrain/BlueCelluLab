:Comment :
:Reference :A quantitative description of membrane current and its application   conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952)

NEURON	{
	SUFFIX HHK
	USEION k READ ek WRITE ik
	RANGE gHHKbar, gHHK, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gHHKbar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gHHK	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gHHK = gHHKbar*m*m*m*m
	ik = gHHK*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
        if(v == 10){
            v = v+0.0001
        }
		mAlpha = (0.01*(10-v))/(exp((10-v)/10) - 1.0)
		mBeta  = 0.125 * (exp(-v/80))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
