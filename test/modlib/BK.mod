: BK-type Purkinje calcium-activated potassium current
: Created 8/19/02 - nwg

NEURON {
       SUFFIX BK
       USEION k READ ek WRITE ik
       USEION ca READ cai
       RANGE gBKbar, gBK, ik
}

UNITS {
      (mV) = (millivolt)
      (mA) = (milliamp)
      (mM) = (milli/liter)
}

PARAMETER {
          v            (mV)
          gBKbar = .007 (mho/cm2)
          zTau = 1              (ms)
          ek           (mV)
          cai          (mM)
}

ASSIGNED {
         :mInf
         :mTau          (ms)
         :hInf
         :hTau          (ms)
         zInf
         ik            (mA/cm2)
         gBK	       (S/cm2)
}

STATE {
      :m   FROM 0 TO 1
      z   FROM 0 TO 1
      :h   FROM 0 TO 1
}

BREAKPOINT {
           SOLVE states METHOD cnexp
           :gBK  = gBKbar * m * m * m * z * z * h
           :HACKED
           gBK  = gBKbar * z * z 
           ik   =  gBK * (v - ek)
}

DERIVATIVE states {
        rates(v,cai)
        :m' = (mInf - m) / mTau
        :h' = (hInf - h) / hTau
        z' = (zInf - z) / zTau
}

PROCEDURE rates(Vm (mV), ca(mM)) {
          LOCAL v
          v = Vm + 5
          :mInf = 1 / (1 + exp(-(v - (-28.9)) / 6.2))
          :mTau = (1e3) * (0.000505 + 1/(exp((v+ -33.3)/(-10)) + exp((v+ 86.4)/10.1)))
          if(ca < 1e-7){
	              ca = ca + 1e-07
          }
          zInf = 1/(1 + 0.001 / ca)
          :hInf = 0.085 + (1- 0.085) / (1+exp((v - -32)/5.8))
          :hTau = (1e3) * (0.0019 + 1/(exp((v + -54.2)/(-12.9))+exp((v+ 48.5)/5.2)))
}

INITIAL {
        rates(v,cai)
        :m = mInf
        z = zInf
        :h = hInf
}
