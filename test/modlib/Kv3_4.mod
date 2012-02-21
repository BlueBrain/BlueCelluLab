:Comment :
:Reference : :		Cloning of ShIII (Shaw-like) cDNAs encoding a novel high-voltage-activating, TEA sensitive, type-A K+ channel, Proc. R. Soc. Lond. B, 248:9-18, (1992)

NEURON	{
	SUFFIX Kv3_4
	USEION k READ ek WRITE ik
	RANGE gKv3_4bar, gKv3_4, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv3_4bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv3_4	(S/cm2)
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
	gKv3_4 = gKv3_4bar*m*h
	ik = gKv3_4*(v-ek)
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
		mInf =  1/(1+exp(((v -(-3.400))/(-8.400))))
		mTau =  10.000/(1+exp(((v -(4.440))/(38.140))))
		hInf =  1/(1+exp(((v -(-53.320))/(7.400))))
		hTau =  20000.000/(1+exp(((v -(-46.560))/(-44.140))))
	UNITSON
}
