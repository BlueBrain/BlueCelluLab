:Comment :
:Reference : :		Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000

NEURON	{
	SUFFIX Kslow
	USEION k READ ek WRITE ik
	RANGE gKslowbar, gKslow, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKslowbar = 0.00001 (S/cm2) 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKslow	(S/cm2)
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
	gKslow = gKslowbar*m*m*h
	ik = gKslow*(v-ek)
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
		mInf =  (1/(1 + exp(-(v+14)/14.6))) 
        if(v<-50){
		    mTau =  (1.25+175.03*exp(-v * -0.026))
        }else{
            mTau = (1.25+13*exp(-v*0.026))
        }
		hInf =  1/(1 + exp(-(v+54)/-11))
		hTau =  360+(1010+24*(v+55))*exp(-((v+75)/48)^2)
	UNITSON
}
