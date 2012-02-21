:Comment :
:Reference : :		Sequence and functional expression in Xenopus oocytes of a human insulinoma and islet potassium channel n  Philipson L.H.  Proc. Natl. Acad. Sci. U.S.A. 88:53-57(1991).

NEURON	{
	SUFFIX Kv1_5
	USEION k READ ek WRITE ik
	RANGE gKv1_5bar, gKv1_5, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv1_5bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv1_5	(S/cm2)
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
	gKv1_5 = gKv1_5bar*m*h
	ik = gKv1_5*(v-ek)
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
		mInf =  1.0000/(1+ exp((v - -6.0000)/-6.4000))
		mTau =  (-0.1163 * v) + 8.3300
		hInf =  1.0000/(1+ exp((v - -25.3000)/3.5000))
		hTau =  (-15.5000 * v) + 1620.0000
	UNITSON
}
