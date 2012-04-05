:Comment :
:Reference :A quantitative description of membrane current and its application   conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952)
:Naundorf HHK channel

NEURON	{
	SUFFIX HHK3
	USEION k READ ek WRITE ik
	RANGE gmax, g, ik 
	RANGE mInf
	RANGE mTau
	GLOBAL k_inf, d_inf, A_inf, k_tau, d_tau, A_tau

}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gmax = 0.036 (S/cm2) 
	k_inf = -0.07992
	d_inf = -55
	A_inf = 1
	k_tau = -0.0125
	d_tau = -65
	A_tau = 0.125
	
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gmax*m
	ik = g*(v-ek)
}

DERIVATIVE states	{
	rates(v)
	m' = (mInf-m)/mTau
}

INITIAL{
	rates(v)
	m = mInf
}

PROCEDURE rates(v){
	:A / ( 1 + exp( -k*(d-v) ))
	mInf = A_inf/(1 + exp(-k_inf*(d_inf - v)))
	:A * exp( -k*(d-v) )
      	mTau = A_tau*exp(k_tau*(v - d_tau))
}


