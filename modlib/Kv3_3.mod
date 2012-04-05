:Comment :
:Reference : :		The contribution of Dendritic Kv3 K+ channels to burst threshold in a sensory neuron, J.Neurosci., 21(1),125-135 (1992)

NEURON	{
	SUFFIX Kv3_3
	USEION k READ ek WRITE ik
	RANGE gKv3_3bar, gKv3_3, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv3_3bar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv3_3	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gKv3_3 = gKv3_3bar*m
	ik = gKv3_3*(v-ek)
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
		mInf =  1/(1+exp(((v -(18.700))/(-9.700))))
		mTau =  20.000/(1+exp(((v -(-46.560))/(-44.140))))
	UNITSON
}
