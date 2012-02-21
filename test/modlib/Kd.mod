:Generic delayed channel

NEURON  {
        SUFFIX Kd
        USEION k READ ek WRITE ik
        RANGE gKdbar, gKd, ik, vshift 
}

UNITS   {
        (S) = (siemens)
        (mV) = (millivolt)
        (mA) = (milliamp)
}

PARAMETER       {
        gKdbar = 0.00001 (S/cm2)
        vshift = 0 (mV)
}

ASSIGNED        {
        v       (mV)
        ek      (mV)
        ik      (mA/cm2)
        gKd     (S/cm2)
        mInf
        mTau
        hInf
        hTau
}

STATE   {
        m
        h
}

BREAKPOINT      {
        SOLVE states METHOD cnexp
        gKd = gKdbar*(m^3)*h
        ik = gKd*(v-ek)
}

DERIVATIVE states       {
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
                v = v + vshift
                mInf =  1/(1 + exp(-(v+50)/20))
                mTau =  2
                hInf =  1/(1 + exp(-(v+70)/-6))
                hTau =  150
                v = v - vshift
        UNITSON
} 
