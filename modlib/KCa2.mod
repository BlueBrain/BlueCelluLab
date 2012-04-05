:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 349 $]
:[$Date: 2007-05-08 16:31:38 +0200 (Tue, 08 May 2007) $]
:[$Author: rajnish $]
:Comment :
:Reference : :		Baccus,PNAS(1998);95;8345-8350

NEURON	{
	SUFFIX KCa2_0262
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gKCa2bar, gKCa2, ik, BBiD 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKCa2bar = 0.00001 (S/cm2) 
	BBiD = 262 
	cai          (mM)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKCa2	(S/cm2)
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
	gKCa2 = gKCa2bar*m
	ik = gKCa2*(v-ek)
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
		mAlpha = 0.1 * (cai/ 0.01)
		mBeta = 0.1
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
