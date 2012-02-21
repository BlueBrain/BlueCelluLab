:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 349 $]
:[$Date: 2007-05-08 16:31:38 +0200 (Tue, 08 May 2007) $]
:[$Author: rajnish $]
:Comment :
:Reference : :		Reference: Davison et. al. J Neurophysiol(2003) 90:1921-1935, with model parameters taken from Bhalla and Bower, J. Neurophysiol(1993),69:1948-1983

NEURON	{
	SUFFIX KCa3_0263
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gKCa3bar, gKCa3, ik, BBiD 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKCa3bar = 0.00001 (S/cm2) 
	BBiD = 263 
	cai          (mM)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKCa3	(S/cm2)
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
	gKCa3 = gKCa3bar*m
	ik = gKCa3*(v-ek)
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
		mAlpha = 500(/ms)*(0.015-cai*1(/mM))/(exp((0.015-cai*1(/mM))/0.0013)-1)
		mBeta = 0.05
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
