:Comment :
:Reference :A. Bibbig et.al. J. Neurosci. Nov. 2001,21(22):9053-9067

NEURON	{
	SUFFIX km1
	USEION k READ ek WRITE ik
	RANGE gkm1bar, gkm1, ik , mInf, mTau
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gkm1bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gkm1	(S/cm2)
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
	gkm1 = gkm1bar*m
	ik = gkm1*(v-ek)
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
		mAlpha = 0.02/(1+exp((40-v)/5))
		mBeta  = 0.01*exp((17-v)/18)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
