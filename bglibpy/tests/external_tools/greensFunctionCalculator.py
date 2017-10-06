import numpy as np
import math
import scipy.optimize

from sys import argv, stdout
import os
import time

# auxiliary tanh function because the numpy one is bad
def tanh(x):
    return (1.-np.exp(-2.*x))/(1.+np.exp(-2.*x))    
# product of two tanh functions, for stability
def tanhtanh(x1,x2):
    return (1. + np.exp(-2.*(x1+x2)) - np.exp(-2.*x1) - np.exp(-2.*x2))/(1. + np.exp(-2.*(x1+x2)) + np.exp(-2.*x1) + np.exp(-2.*x2))
# one minus tanhtanh, for stability
def one_minus_tanhtanh(x1,x2):
    return 2.*(np.exp(-2*x1) + np.exp(-2*x2))/(1. + np.exp(-2.*(x1+x2)) + np.exp(-2.*x1) + np.exp(-2.*x2))
# traps for zero in denominators of rate equations    
def vtrap(x, y):
	if np.abs(x/y) < 1e-6:
		trap = y*(1. - x/(y*2.))
	else:
		trap = x/(np.exp(x/y) - 1.)
	return trap
# HH kinetic rates
alphan  = lambda V:  .01 * vtrap(-(V+55.),10.)          *1.0e3
betan   = lambda V:  .125* np.exp(-(V+65.)/80.)         *1.0e3
alpham  = lambda V:  .1  * vtrap(-(V+40.),10.)          *1.0e3
betam   = lambda V: 4.   * np.exp(-(V+65.)/18.)         *1.0e3
alphah  = lambda V:  .07 * np.exp(-(V+65.)/20.)         *1.0e3
betah   = lambda V: 1.   / (np.exp(-(V+35.)/10.) + 1.)  *1.0e3
# derivatives of HH kinetic rates
alphandot  = lambda V:  -0.01*(np.exp(-(V+55.)/10.) - 1. + (V+55.)*np.exp(-(V+55.)/10.)/10.)/np.power(np.exp(-(V+55.)/10.)-1.,2)    *1.0e3
betandot   = lambda V:  -0.125*np.exp(-(V+65.)/80.)/80.                                                                             *1.0e3
alphamdot  = lambda V:  -0.1* (np.exp(-(V+40.)/10.) - 1. + (V+40.)*np.exp(-(V+40.)/10.)/10.)/np.power(np.exp(-(V+40.)/10.)-1.,2)    *1.0e3
betamdot   = lambda V:  -4./10. * np.exp(-(V+65.)/10.)                                                                              *1.0e3
alphahdot  = lambda V:  -0.07/20. * np.exp(-(V+65.)/20.)                                                                            *1.0e3
betahdot   = lambda V:  0.1*np.exp(-(V+35.)/10.)/np.power(np.exp(-(V+35.)/10.)+1.,2)                                                *1.0e3



class greensFunctionCalculator:
    def __init__(self, dend_diams = np.array([0.45, 1.]), dend_lengths = np.array([937.5,  559.016994375]), 
                    somaL = 25., somaDiam = 25., C_m = 0.8, g_m = 2e-5, g = 2e-5, r_a = 100., readconfig = 1, HH=1, conffilename = 'ball2stick.cfg'):
        '''
        Constructor used to initialize the necessary parameters of the neuron model.
        The neuron is passive, has N = len(dend_diams) = len(dend_lengths) dendrites, 
        a membrane capacitance that is uniform throughout, a membrane conductance that
        can be specified for the dendrites and for the soma and an intracellular 
        resistance parameter.
        
        -input:
            [dend_diams]: radii of the dendrites connected to soma [um]
            [dend_lengths]: lengths of the dendrites connected to soma [um]
            [somaL]: length of soma (NEURON convention) [um]
            [somaDiam]: diameter of soma (NEURON convention) [um]
            [C_m]: membrane capacitance [uF/cm^2]
            [g_m]: membrane conductance in dendrites [S/cm^2]
            [g]: membrane conductance in soma [S/cm^2]
            [r_a]: intracellular resistance [ohm*cm]
        '''
        self.lengths = dend_lengths # um
        self.diams = dend_diams/2.		# um
        self.C_m = C_m				# uF / cm^2
        self.somaL = somaL			# um
        self.somaDiam = somaDiam	# um
        self.g_s = g			    # S / cm^2
        self.r_a = r_a				# ohm * cm
        self.g_m = g_m		        # 1 / (ohm * cm^2)
        self.HH = HH
        
        if readconfig == 1:
            from ConfigParser import ConfigParser
            conffile = conffilename
            fid = open(conffile,'r')
            config = ConfigParser()
            config.readfp(fid)
            fid.close()
            
            CM = config.getfloat('neuron', 'CM')
            RM = config.getfloat('neuron', 'RM')
            RA = config.getfloat('neuron', 'RA')
            EL = config.getfloat('neuron', 'EL')
            
            somaD = config.getfloat('soma', 'D')
            somaL = config.getfloat('soma', 'L')
            
            import json
            dendlengths = np.array(json.loads(config.get('morph', 'lengths')))
            denddiams = np.array(json.loads(config.get('morph', 'diams')))
            
            self.lengths = dendlengths  # um
            self.diams = denddiams/2.	# um
            self.C_m = CM				# uF / cm^2
            self.somaL = somaL			# um
            self.somaDiam = somaD	    # um
            self.g_s = 1./RM			    # S / cm^2
            self.r_a = RA				# ohm * cm
            self.g_m = 1./RM            # 1 / (ohm * cm^2)


        # convert lengths to cm
        self.lengths = self.lengths/1e4
        self.diams = self.diams/1e4
        self.somaL = self.somaL/1e4
        self.somaDiam = self.somaDiam/1e4
        # convert resistances to mega ohms (conductances in micro siemens)
        self.r_a = self.r_a/1e6
        self.g_m = self.g_m*1e6
        self.g_s = self.g_s*1e6
        # soma surface 
        somaA = math.pi * self.somaDiam * self.somaL 
        
        if HH == 1:
            import explicitIOD as ex
            HHneuron = ex.HHneuron(HH=1)
            Vm = HHneuron.run_syn([], tdur=200, syndends=[], synlocs = [], syntaus = [], synweights = np.array([]))
            E_eq = Vm['vmsoma'][-1]  # mv
            # params
            self.EL = -54.3      	# v_init
            self.gL = 0.0003*somaA*1e6	# uS/cm2
            self.EK = -77        	# mV
            self.gK = 0.036*somaA*1e6 	# uS/cm2
            self.ENa = 50        	# mV
            self.gNa = 0.120*somaA*1e6 	# uS/cm2
            # state variables sv = [minf, hinf, ninf]
            alphain = np.array([.1 * vtrap(-(E_eq+40.),10.), .07 * np.exp(-(E_eq+65.)/20.), .01*vtrap(-(E_eq+55.),10.) ])
            betain = np.array([4. * np.exp(-(E_eq+65.)/18.), 1. / (np.exp(-(E_eq+35.)/10.) + 1.), .125*np.exp(-(E_eq+65.)/80.)])
            sv = alphain/(alphain + betain)
            self.g_s = (self.gNa*sv[0]*sv[0]*sv[0]*sv[1] + self.gK*sv[2]*sv[2]*sv[2]*sv[2] + self.gL)/somaA # S/cm^2
            self.E_eq = E_eq
        else:
            self.E_eq = EL

        # calculate soma conductance (uS) and capacitance (uF)
        self.somaC = self.C_m*somaA
        self.somaG = self.g_s*somaA
        
    def getSomaKernel(self, N = np.power(2,9), dt = 0.1*1e-2):
        tmax = N*dt
        t = np.arange(0.,N*dt,dt)
        oneovertau = self.somaG/(self.somaC) 
        return np.exp(-oneovertau*t), t 
        
    def calc_somaImpedance(self, s, E):
        if self.HH == 1:
            m = alpham(E)/(alpham(E) + betam(E))
            n = alphan(E)/(alphan(E) + betan(E))
            h = alphah(E)/(alphah(E) + betah(E))
            z_C = 1./(self.somaC*s)
            z_L = -1./self.gL
            z_K1 = -1./(self.gK*n*n*n*n)
            z_K2 = (s + alphan(E) + betan(E))/(4.*self.gK*n*n*n*(self.EK-E)*(alphandot(E)-n*(alphandot(E)+betandot(E))))
            z_Na1 = -1./(self.gNa*m*m*m*h) 
            z_Na2 = (s + alpham(E) + betam(E))/(3*self.gNa*m*m*h*(self.ENa-E)*(alphamdot(E)-m*(alphamdot(E)+betamdot(E))))
            z_Na3 = (s + alphah(E) + betah(E))/(self.gNa*m*m*m*(self.ENa-E)*(alphahdot(E)-h*(alphahdot(E)+betahdot(E))))
            z_soma = 1./(1./z_C - 1./z_K2 - 1./z_L - 1./z_K1 - 1./z_Na1 - 1./z_Na2 - 1./z_Na3)
            #~ temp1 = -np.log(z_L*z_K1*z_K2*z_Na1*z_Na2*z_Na3 - z_C*z_K1*z_K2*z_Na1*z_Na2*z_Na3 \
                            #~ - z_C*z_L*z_K2*z_Na1*z_Na2*z_Na3 - z_C*z_L*z_K1*z_Na1*z_Na2*z_Na3 \
                            #~ - z_C*z_L*z_K1*z_K2*z_Na2*z_Na3 - z_C*z_L*z_K1*z_K2*z_Na1*z_Na3 \
                            #~ - z_C*z_L*z_K1*z_K2*z_Na1*z_Na2)
            #~ temp2 = np.log(z_C*z_L*z_K1*z_K2*z_Na1*z_Na2*z_Na3)
            #~ z_soma = np.exp(temp1+temp2)
            #~ import matplotlib.pyplot as pl
            #~ pl.plot(s.imag,z_soma.real,'b')
            #~ pl.plot(s.imag,z_soma.imag,'r')
            #~ pl.show()
        else:
            z_soma = 1./(self.somaG + self.somaC * s)
        return z_soma
        
    def greensFunctionFrequency(self, synapsedendrite, synapseloc, samplefreqs):
        '''
        Computes the input impedance, the transfer impedance from synapse
        to soma and the derivative of the transfor impedance for the neuron
        morphology specified in the constructor, where the synapse is located
        on [synapsedendite], for a given range of frequencies
        
        -input:
            [synapsedendrite]: integer specifying the dendrite on which the
                    synapse is located
            [synapseloc]: floating point number specifying the distance from
                    soma to synapse, measured in um
            [samplefreqs]: numpy array of complex numbers at which the greens-
                    function should be evaluated
            [nosomaleak]: flag: 0 means no soma leak current and 1 means soma
                    has leak current
        
        -output:
            [G_synsyn]: the input impedance
            [G_synsoma]: the transfer impedance from synapse to soma
            [Gdot_synsoma]: when transformed to the time-domain, gives
                the derivative of the transfer impedance from synapse to
                soma
        '''
        synseg = synapsedendrite
        inds = np.array(range(len(self.diams)))
        nosynseg = np.where(inds!=synseg)[0]
        L = self.lengths[synseg]
        D = synapseloc*L 
        s = samplefreqs
        
        
        # original soma impedance
        z_soma = self.calc_somaImpedance(s,self.E_eq)
        z_snew = z_soma

        ## STEP 0: params of nonsynaptic dentrites
        if nosynseg.size:
            for j in range(len(nosynseg)):
                i = nosynseg[j]
                # intracellullar impedance
                z_a = self.r_a/(math.pi*self.diams[i]*self.diams[i])
                # transmembrane impedance
                z_m = 1./(2.*math.pi*self.diams[i]*(s*self.C_m  + self.g_m))
                # space constant
                gamma = np.sqrt(z_a/z_m)
                # characteristic impedance
                z_c = z_a/gamma
                # modified soma impedance
                z_snew = 1. / ((1./z_snew) + (1./z_c) * np.tanh(gamma*self.lengths[i]))

        ## STEP 1: params of synaptic dendrite
        i = synapsedendrite
        # intracellullar impedance
        z_a = self.r_a/(math.pi*self.diams[i]*self.diams[i])
        # transmembrane impedance
        z_m = 1./(2*math.pi*self.diams[i]*(s*self.C_m + self.g_m))
        # space constant
        gamma = np.sqrt(z_a/z_m)
        # characteristic impedance
        z_c = z_a/gamma

        ## STEP 3: greens functions
        # pulse-voltage response at synapse
        G_synsyn = np.exp(2.*np.log(z_c) + 2.*gamma*D) * np.power((1.+np.exp(-2.*gamma*D))/2.,2) \
                        * (tanh(gamma*D) + (z_snew/z_c)) \
                        * one_minus_tanhtanh(gamma*D, gamma*L) / (z_c + z_snew*tanh(gamma*L))
        # pulse-voltage response at soma
        G_synsoma = z_c * z_snew * np.cosh(gamma*D) * \
                        one_minus_tanhtanh(gamma*D, gamma*L) / (z_c + z_snew*tanh(gamma*L))
        # pulse-current response at soma
        GI_synsoma = (1./z_a) * z_c * z_c * gamma * np.cosh(gamma*D) * \
                        one_minus_tanhtanh(gamma*D, gamma*L) / (z_c + z_snew*tanh(gamma*L))
        # pulse-current response in other branches
        if nosynseg.size:
            GI_awaysoma = G_synsoma/z_soma - GI_synsoma
        else:
            GI_awaysoma = np.zeros(len(GI_synsoma))
        
        #~ import matplotlib.pyplot as pl
        #~ pl.plot(s.imag, G_synsoma,'b')
        #~ pl.plot(s.imag, G_synsyn,'r')
        #~ pl.plot(s.imag, GI_synsoma,'g')
        #~ pl.show()
        
        
        #~ import scipy.io
        #~ scipy.io.savemat('temp.mat', mdict={'s':s.real, 'G_synsyn': G_synsyn})
        
        return G_synsyn, G_synsoma, GI_synsoma, GI_awaysoma
        
    def greensFunctionTime(self, synapsedendrite = 0, synapseloc = 0.5, N = np.power(2,9), dt = 0.1*1e-2, test = 0, tmax=0.3):
        '''
        Computes the Greens function (pulse response) on the input location and 
        at the soma, and the derivative of the Greens Function at the soma.
        
        -input:
            [synapsedendrite]: integer specifying the dendrite on which the
                    synapse is located
            [synapseloc]: floating point number specifying the distance from
                    soma to synapse, measured in um
            [N]: number of points at which to sample the greens function,
                    should be a power of two
            [dt]: timestep at which the greens function is sampled
            [test]: if 1, print some values that allow for evalutation of the 
                    quality of the transform
            [nosomaleak]: flag: 0 means no soma leak current and 1 means soma
                    has leak current
        
        -output:
            [Gt_synsyn]: the input greens function
            [Gt_synsoma]: the transfer greens function from synapse to soma
            [Gtdot_synsoma]: the derivative of the transfer greens function
            [t]: array of times at which the greens functions are evaluated
        '''
        # setting parameters
        smax = 2*math.pi/(2*dt)
        ds = 2*math.pi/(2*N*dt)
        samplefreqs = np.arange(-smax,smax,ds)*1j
        t = np.arange(0.,N*dt,dt)
        # calc greens function
        G_synsyn, G_synsoma, GI_synsoma, GI_awaysoma = self.greensFunctionFrequency(synapsedendrite, synapseloc, samplefreqs)
        Gt_synsyn, t = self.calcFFT(G_synsyn, N, dt, test)
        Gt_synsoma, t = self.calcFFT(G_synsoma, N, dt, test)
        GIt_synsoma, t = self.calcFFT(GI_synsoma, N, dt, test)
        GIt_awaysoma, t = self.calcFFT(GI_awaysoma, N, dt, test)
        # keep relevant parts
        Gt_synsyn = Gt_synsyn[0:tmax/dt]
        Gt_synsoma = Gt_synsoma[0:tmax/dt]
        GIt_synsoma = GIt_synsoma[0:tmax/dt]
        GIt_awaysoma = GIt_awaysoma[0:tmax/dt]
        t = t[0:tmax/dt]
        
        import matplotlib.pyplot as pl
        #pl.plot(t,Gt_synsoma,'b')
        #pl.plot(t,Gt_synsyn,'r')
        #pl.plot(t,GIt_synsoma + GIt_awaysoma, 'g')
        #pl.show()
        
        return np.array(Gt_synsyn)/1000., np.array(Gt_synsoma)/1000., np.array(GIt_synsoma)/1000., np.array(GIt_awaysoma)/1000., t*1000
        
    def calcFFT(self, arr, N, dt, test=0, method='hannig'):
        
        if method == 'standard':
            smax = math.pi/(dt)
            ds = math.pi/(N*dt)
            samplefreqs = np.arange(-smax,smax,ds)*1j
            #fft
            t = np.arange(0.,N*dt,dt)
            arr = np.array(arr)
            fftarr = np.fft.ifft(arr)
            fftarr = fftarr[0:len(fftarr)/2]
            # scale factor
            scale = 2*N*ds*np.exp(-1j*t*smax)/(2*math.pi)
            fftarr = scale*fftarr
        elif method == 'trapezoid':
            smax = math.pi/(dt)
            ds = math.pi/(N*dt)
            samplefreqs = np.arange(-smax,smax,ds)*1j
            #fft
            t = np.arange(0.,N*dt,dt)
            arr = np.array(arr)
            fftarr = np.fft.ifft(arr)
            fftarr = fftarr[0:len(fftarr)/2]
            # scale factor
            prefactor = (1.-np.cos(ds*t))/(ds*np.power(t,2))
            prefactor[0] = 0.5*ds
            scale = 4.*N*prefactor*np.exp(-1j*t*smax)/(2*math.pi)
            fftarr = scale*fftarr
        elif method == 'hannig':
            smax = math.pi/(dt)
            ds = math.pi/(N*dt)
            samplefreqs = np.arange(-smax,smax,ds)*1j
            #fft
            t = np.arange(0.,N*dt,dt)
            window = 0.5*(1.+np.cos(math.pi*samplefreqs.imag/smax))
            arr = np.array(arr*window)
            fftarr = np.fft.ifft(arr)
            fftarr = fftarr[0:len(fftarr)/2]
            # scale factor
            scale = 2*N*ds*np.exp(-1j*t*smax)/(2*math.pi)
            fftarr = scale*fftarr
        elif method == 'simpson':
            ds = math.pi/(N*dt)
            smax = math.pi/dt
            samplefreqs = np.arange(-smax,smax,ds)*1j
            #fft
            t = np.arange(0.,N*dt,dt)
            arreven = arr[::2]
            arrodd = arr[1::2]
            arreventrans = np.fft.ifft(arreven)
            arroddtrans = np.fft.ifft(arrodd)
            fftarr = ds/(6.*math.pi)*(-arr[0]*np.exp(-1j*smax*t) + 2.*N*np.exp(-1j*smax*t)*arreventrans \
					+4.*N*np.exp(-1j*(smax-ds)*t)*arroddtrans + arr[-1]*np.exp(1j*smax*t))
        elif method == 'quadrature':
            import scipy.integrate as integ
            t = np.arange(0.,N*dt,dt)
            fftarr = np.zeros(N)
            smax = 1000.
            for i in range(len(t)):
                fftarr[i] = integ.romberg(self.aux_func,-smax,smax,args = [t[i]])
        
        #~ fftarr[0] = sum((arr[0:-1]+arr[1:])/2.)*(ds)/(2*math.pi)
        if test==1:
            print 'Tests:'
            print 'Int: ', sum((arr[0:-1]+arr[1:])/2.)*(ds)/(2*math.pi), ', Trans zero freq: ', fftarr[0]
            print 'G zero_freq: ', arr[len(samplefreqs)/2].real, ', G integral: ', (fftarr[0].real/2. + sum(fftarr[1:-1].real) + fftarr[-1].real/2.)*dt
        
        return fftarr.real, t
        
    def aux_func(self, s, t):
        return (self.greensFunctionFrequency(0, 0.95, np.array([s]))[0] * np.exp(1j*s*t))[0]
        
    #~ def calcDFT(self, arr, N, dt):
        #~ import scipy.integrate as integ
        #~ functransint = np.zeros(len(k))
#~ 
        #~ for i in range(len(k)):
            #~ arrtransint[i] = integ.romberg(self.fourrierfunc,-20.,20.,args = [k[i]])
        #~ pl.plot(k,functransint, 'g', label='numerical int evaluation')

    def trapezoidalCorrection(self, times):
        W = 2.*(1.-np.cos(times))/np.power(times,2)
        alpha0 = -(1.-np.cos(times))/np.power(times,2) + 1j*(times-np.cos(times))/np.power(times,2)
        #~ W = 1.-(1./12.)*np.power(times,2)+(1./360.)*np.power(times,4)-(1./20160.)*np.power(times,6)
        #~ alpha0 = -(1./2.)+(1./24.)*np.power(times,2)-(1./720.)*np.power(times,4)+(1./40320.)*np.power(times,6) \
                #~ + 1j*times*((1./6.)-(1./120.)*np.power(times,2)+(1./5040.)*np.power(times,4)-(1./362880.)*np.power(times,6))
        return W, alpha0


class pneuron:
    def __init__(self, numsynapses, greensFunctions_synsoma, greensFunctions_synsyn, tgreensFunction, 
                    taus = [1.5,1.5], weights = np.array([0.001,0.0004]), E_eq=-70., E_r = [0.,0.], readconfig=1, conffilename = 'ball2stick.cfg'):
        '''
        Constructor to initialize the necessary parameters and kernels to run the
        point neuron model with transfer greens functions.
        
        -input:
            [numsynapses]: the total number of synapses
            [greensFunctions_synsoma]: list of arrays of equal length that contain
                the different transfer greens functions
            [greensFunctions_synsyn]: list of arrays of equal length that contain
                the different input greens functions
            [tgreensFunction]: the array of times at which the greens functions
                are evaluated [ms]
            [taus]: array of decay times of the the synaptic conductances [ms]
            [weigths]: array of synaptic weights [uS]
            [E_eq]: equilibrium potential of the neuron model 
        '''
        self.tgreensFunction = tgreensFunction
        self.numsynapses = numsynapses
        self.greensFunctions_synsoma = greensFunctions_synsoma 
        self.greensFunctions_synsyn = greensFunctions_synsyn 
        self.V_soma = E_eq      # mV
        self.V_syns = E_eq*np.ones(numsynapses)
        self.E_eq = E_eq
        self.E_r = E_r
        if readconfig == 1:
            from ConfigParser import ConfigParser
            conffile = conffilename
            fid = open(conffile,'r')
            config = ConfigParser()
            config.readfp(fid)
            fid.close()
            EL = config.getfloat('neuron', 'EL')
            self.E_eq = EL
        # set synapse params
        self.taus = taus        # ms
        self.weight =  weights  # uS
        
    def run_syn(self, tmax, dt, spiketimes):
        '''
        Runs the point neuron model with the parameters specified in the constructor
        for a duration tmax
        
        -input:
            [tmax]: duration of the simulation [ms]
            [dt]: timestep of the simulation [ms]
            [spiketimes]: list of input spiketimes for each synapse [ms]
        
        -output:
            [Vsoma]: soma voltage trace
            [V_syns]: list of synaptic voltage traces
            [time]: corresponding time values
        '''
        timesim = np.arange(0.,tmax,dt)
        tsim = np.arange(0., self.tgreensFunction[-1],dt)
        # interpolate at the desired sampling frequency (timestep of the simulation)
        G_synsoma = []
        G_synsyn = []
        for j in range(self.numsynapses):
            G_synsoma.append(np.interp(tsim, self.tgreensFunction, self.greensFunctions_synsoma[j,:]))
            G_synsyn.append(np.interp(tsim, self.tgreensFunction, self.greensFunctions_synsyn[j,:]))
        G_synsyn = np.array(G_synsyn)
        G_synsoma = np.array(G_synsoma)
        N = len(tsim)
        # initialize voltage and conductance arrays
        V_soma = self.E_eq * np.ones(int(tmax/dt) + N)
        V_syns = np.zeros((self.numsynapses, int(tmax/dt)+N))
        B = np.zeros(self.numsynapses)        
        synwindow = np.zeros((self.numsynapses, int(tmax/dt)+N))
        # bookkeeping parameter
        k = np.zeros(self.numsynapses, dtype=int)
        # auxiliary storage array
        I = np.zeros(self.numsynapses)
        
        stdout.write('>>> Simulating the SRM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()
        start = time.time()
        
        # main computational loop
        for l in range(int(tmax/dt)):
            i = l + N
            # loop over all synapses
            for j in range(self.numsynapses):
                # spike injector
                if k[j] < len(spiketimes[j]) and l*dt <= spiketimes[j][k[j]] and spiketimes[j][k[j]] < (l+1)*dt:
                    B[j] += self.weight[j]
                    k[j] += 1
                B[j] = B[j] - dt*B[j]/self.taus[j]
                synwindow[j,i] = B[j]
                # advance the synaptic voltage traces
                V_syns[j,i] = (self.E_eq - dt * np.sum(G_synsyn[j,1:-1][::-1]*(V_syns[j,i-N+1:i-1]-self.E_r[j])*synwindow[j,i-N+1:i-1]) + \
                            - dt*G_synsyn[j,-1]*(V_syns[j,i-N]-self.E_r[j])*synwindow[j,i-N]/2.) # / (1. - dt*G_synsyn[j,0]*synwindow[j,i]/2.)
                # advance soma inputs            
                I[j] = - (G_synsoma[j,0] \
                        * (V_syns[j,i]-self.E_r[j])*synwindow[j,i]/2. + np.sum(G_synsoma[j,1:-1][::-1] \
                        * (V_syns[j,i-N+1:i-1]-self.E_r[j]) * synwindow[j,i-N+1:i-1]) + G_synsoma[j,-1] \
                        * (V_syns[j,i-N]-self.E_r[j])*synwindow[j,i-N]/2.)
            # advance soma voltage    
            V_soma[i] = self.E_eq + dt*np.sum(I)
            
        stop = time.time()
        stdout.write('>>> Elapsed time: ' + str(int(stop-start)) + ' seconds. <<<\n \n')
        stdout.flush()
            
        return V_soma[N:], V_syns[:,N:], timesim
    
    def run_IC(self, tmax, dt, stimdur, stimamp):
        '''
        Runs the point neuron model with the parameters specified in the constructor
        for a duration tmax
        
        -input:
            [tmax]: duration of the simulation [ms]
            [dt]: timestep of the simulation [ms]
            [spiketimes]: list of input spiketimes for each synapse [ms]
        
        -output:
            [Vsoma]: soma voltage trace
            [V_syns]: list of synaptic voltage traces
            [time]: corresponding time values
        '''
        timesim = np.arange(0.,tmax,dt)
        tsim = np.arange(0., self.tgreensFunction[-1],dt)
        # interpolate at the desired sampling frequency (timestep of the simulation)
        G_synsoma = []
        G_synsyn = []
        for j in range(self.numsynapses):
            G_synsoma.append(np.interp(tsim, self.tgreensFunction, self.greensFunctions_synsoma[j,:]))
            G_synsyn.append(np.interp(tsim, self.tgreensFunction, self.greensFunctions_synsyn[j,:]))
        G_synsyn = np.array(G_synsyn)
        G_synsoma = np.array(G_synsoma)
        N = len(tsim)
        # initialize voltage and conductance arrays
        V_soma = self.E_eq * np.ones(int(tmax/dt) + N)
        V_IC = self.E_eq * np.ones(int(tmax/dt)+N)
        # IC injection array
        I = np.zeros(int(tmax/dt) + N)
        I[N+1:N+int(stimdur/dt)+1] = stimamp * np.ones(int(stimdur/dt))
        
        stdout.write('>>> Simulating the SRM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()
        start = time.time()
        
        # main computational loop
        for l in range(int(tmax/dt)):
            i = l + N
            # advance the synaptic voltage traces
            V_IC[i] = self.E_eq + dt*(G_synsyn[j,0]*I[i]/2. \
                        + np.sum(G_synsyn[j,1:-1][::-1] \
                        * I[i-N+2:i]) + G_synsyn[j,-1] \
                        * I[i-N+1]/2.)
            # advance soma inputs            
            Soma = G_synsoma[j,0] \
                    * I[i]/2. + np.sum(G_synsoma[j,1:-1][::-1] \
                    * I[i-N+2:i]) + G_synsoma[j,-1] \
                    * I[i-N+1]/2.
            # advance soma voltage    
            V_soma[i] = self.E_eq + dt*Soma
            
        stop = time.time()
        stdout.write('>>> Elapsed time: ' + str(int(stop-start)) + ' seconds. <<<\n \n')
        stdout.flush()
            
        return V_soma[N:], V_IC[N:], timesim
        
class aneuron:
    def __init__(self, numsynapses, greensCurrents_synsoma, greensFunctions_synsyn, tgreensFunction, 
                    tau = 1.5, weights = np.array([0.001,0.0004]), E_eq=-70., g_l = 2.e-5,
                    somaL = 25., somaDiam = 25., C_m = 0.8, readconfig = 1):
        '''
        Constructor to initialize the necessary parameters and kernels to run the
        point neuron model with transfer greens functions.
        
        -input:
            [numsynapses]: the total number of synapses
            [greensFunctionsdot_synsoma]: list of arrays of equal length that contain
                the different transfer greens functions
            [greensFunctionsdot_synsyn]: list of arrays of equal length that contain
                the different input greens functions
            [tgreensFunction]: the array of times at which the greens functions
                are evaluated [ms]
            [taus]: array of decay times of the the synaptic conductances [ms]
            [weigths]: array of synaptic weights [uS]
            [E_eq]: equilibrium potential of the neuron model 
        '''
        self.C_m = C_m				# uF / cm^2
        self.somaL = somaL			# um
        self.somaDiam = somaDiam	# um
        self.g = g_l			        # S / cm^2
        self.E_eq = E_eq
        self.HHparams = []
        
        if readconfig == 1:
            from ConfigParser import ConfigParser
            conffile = 'ball2stick.cfg'
            fid = open(conffile,'r')
            config = ConfigParser()
            config.readfp(fid)
            fid.close()
            
            CM = config.getfloat('neuron', 'CM')
            RM = config.getfloat('neuron', 'RM')
            EL = config.getfloat('neuron', 'EL')
            
            somaD = config.getfloat('soma', 'D')
            somaL = config.getfloat('soma', 'L')
            
            import json
            dendlengths = np.array(json.loads(config.get('morph', 'lengths')))
            denddiams = np.array(json.loads(config.get('morph', 'diams')))
            
            self.E_eq = EL              # mV
            self.C_m = CM				# uF / cm^2
            self.somaL = somaL			# um
            self.somaDiam = somaD	    # um
            self.g = 1./RM			    # S / cm^2  

        # convert lengths to cm
        self.somaL = self.somaL/1e4
        self.somaDiam = self.somaDiam/1e4
        # convert resistances to mega ohms (conductances in micro siemens)
        self.g = self.g*1e6
        # calculate soma conductance (uS) and capacitance (uF)
        somaA = 2*math.pi * self.somaDiam/2. * self.somaL 
        self.somaC = self.C_m*somaA
        self.somaG = self.g*somaA
        
        self.tgreensFunction = tgreensFunction
        self.numsynapses = numsynapses
        self.greensCurrents_synsoma = greensCurrents_synsoma 
        self.greensFunctions_synsyn = greensFunctions_synsyn 
        self.V_soma = self.E_eq      # mV
        self.V_syns = self.E_eq*np.ones(numsynapses)
        self.E_eq = self.E_eq
        # set synapse params
        self.taus = tau*np.ones(numsynapses)        # ms
        self.weight =  weights  # uS
        
    def run_syn(self, tmax, dt, spiketimes):
        '''
        Runs the point neuron model with the parameters specified in the constructor
        for a duration tmax
        
        -input:
            [tmax]: duration of the simulation [ms]
            [dt]: timestep of the simulation [ms]
            [spiketimes]: list of input spiketimes for each synapse [ms]
        
        -output:
            [Vsoma]: soma voltage trace
            [V_syns]: list of synaptic voltage traces
            [time]: corresponding time values
        '''
        timesim = np.arange(0.,tmax,dt)
        tsim = np.arange(0., self.tgreensFunction[-1],dt)
        # interpolate at the desired sampling frequency (timestep of the simulation)
        GI_synsoma = []
        G_synsyn = []
        for j in range(self.numsynapses):
            GI_synsoma.append(np.interp(tsim, self.tgreensFunction, self.greensCurrents_synsoma[j,:]))
            G_synsyn.append(np.interp(tsim, self.tgreensFunction, self.greensFunctions_synsyn[j,:]))
        G_synsyn = np.array(G_synsyn)
        GI_synsoma = np.array(GI_synsoma)
        N = len(tsim)
        # initialize voltage and conductance arrays
        V_soma = self.E_eq * np.ones(int(tmax/dt) + N)
        V_syns = self.E_eq * np.ones((self.numsynapses, int(tmax/dt)+N))
        B = np.zeros(self.numsynapses)        
        synwindow = np.zeros((self.numsynapses, int(tmax/dt)+N))
        # bookkeeping parameter
        k = np.zeros(self.numsynapses, dtype=int)
        # auxiliary storage array
        Isoma = np.zeros(self.numsynapses)
        # membrane time constant
        oneovertau = self.somaG/(self.somaC*1000.)  # ms
        oneoverC = 1./(self.somaC*1000.)            # ms
        
        stdout.write('>>> Simulating the GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()
        start = time.time()
        
        # main computational loop
        for l in range(int(tmax/dt)):
            i = l + N
            # loop over all synapses
            for j in range(self.numsynapses):
                # spike injector
                if k[j] < len(spiketimes[j]) and l*dt <= spiketimes[j][k[j]] and spiketimes[j][k[j]] < (l+1)*dt:
                    B[j] += self.weight[j]
                    k[j] += 1
                B[j] = B[j] - dt*B[j]/self.taus[j]
                synwindow[j,i] = B[j]
                # advance the synaptic voltage traces
                V_syns[j,i] = self.E_eq - (dt * np.sum(G_synsyn[j,1:-1][::-1]*V_syns[j,i-N+1:i-1]*synwindow[j,i-N+1:i-1]) + \
                            dt*G_synsyn[j,-1]*V_syns[j,i-N]*synwindow[j,i-N]/2.) / (1. - dt*G_synsyn[j,0]*synwindow[j,i]/2.)
                # advance soma inputs            
                Isoma[j] = - (GI_synsoma[j,0] \
                        * V_syns[j,i]*synwindow[j,i]/2. + np.sum(GI_synsoma[j,1:-1][::-1] \
                        * V_syns[j,i-N+1:i-1] * synwindow[j,i-N+1:i-1]) + GI_synsoma[j,-1] \
                        * V_syns[j,i-N]*synwindow[j,i-N]/2.)
            # advance soma voltage    
            V_soma[i] = V_soma[i-1] - dt*oneovertau*(V_soma[i-1] - self.E_eq) + dt*oneoverC*dt*np.sum(Isoma) 
            
        stop = time.time()
        stdout.write('>>> Elapsed time: ' + str(int(stop-start)) + ' seconds. <<<\n \n')
        stdout.flush()
            
        return V_soma[N:], V_syns[:,N:], timesim

    def run_IC(self, tmax, dt, stimdur, stimamp, HH=0.):
        '''
        Runs the point neuron model with the parameters specified in the constructor
        for a duration tmax
        
        -input:
            [tmax]: duration of the simulation [ms]
            [dt]: timestep of the simulation [ms]
            [spiketimes]: list of input spiketimes for each synapse [ms]
        
        -output:
            [Vsoma]: soma voltage trace
            [V_syns]: list of synaptic voltage traces
            [time]: corresponding time values
        '''
        timesim = np.arange(0.,tmax,dt)
        tsim = np.arange(0., self.tgreensFunction[-1],dt)
        # interpolate at the desired sampling frequency (timestep of the simulation)
        GI_synsoma = []
        G_synsyn = []
        for j in range(self.numsynapses):
            GI_synsoma.append(np.interp(tsim, self.tgreensFunction, self.greensCurrents_synsoma[j,:]))
            G_synsyn.append(np.interp(tsim, self.tgreensFunction, self.greensFunctions_synsyn[j,:]))
        GI_synsoma = np.array(GI_synsoma)
        G_synsyn = np.array(G_synsyn)
        N = len(tsim)
        # initialize voltage and conductance arrays
        V_soma = self.E_eq * np.ones(int(tmax/dt) + N)
        V_IC = self.E_eq * np.ones(int(tmax/dt)+N)
        I_dend = np.zeros(int(tmax/dt)+N)
        # IC injection array
        I = np.zeros(int(tmax/dt) + N)
        I[N+1:N+int(stimdur/dt)+1] = stimamp * np.ones(int(stimdur/dt))
        # membrane time constant
        oneovertau = self.somaG/(self.somaC*1000.)  # ms
        oneoverC = 1./(self.somaC*1000.)            # ms
        
        stdout.write('>>> Simulating the GCM for ' + str(tmax) + ' ms. <<<\n')
        stdout.flush()
        start = time.time()
        
        # main computational loop
        for l in range(int(tmax/dt)):
            i = l + N
            # advance the dendritic voltage traces
            V_IC[i] = self.E_eq + dt*(G_synsyn[j,0]*I[i]/2. \
                        + np.sum(G_synsyn[j,1:-1][::-1] \
                        * I[i-N+2:i]) + G_synsyn[j,-1] \
                        * I[i-N+1]/2.)
            # advance soma inputs            
            Isoma = GI_synsoma[j,0] \
                    * I[i]/2. + np.sum(GI_synsoma[j,1:-1][::-1] \
                    * I[i-N+2:i]) + GI_synsoma[j,-1] \
                    * I[i-N+1]/2.
            # advance soma voltage    
            V_soma[i] = V_soma[i-1] - dt*oneovertau*(V_soma[i-1] - self.E_eq) + dt*oneoverC*dt*Isoma 
            I_dend[i] = dt*Isoma
            
        stop = time.time()
        stdout.write('>>> Elapsed time: ' + str(int(stop-start)) + ' seconds. <<<\n \n')
        stdout.flush()
            
        return V_soma[N:], V_IC[N:], I_dend[N:], timesim 
        
def compute_system(spiketimes, tdur, dt, conffile_name = 'ball2stick.cfg', numsyn=1, syndend = [0], synloc = [0.95], gbar = [0.001], decay = [5.], E_rev = [0.]):
    '''
    Simulate a passive ball + N sticks neuron model, with at most one synapse per stick.
    
    -input:
        spiketimes: list of arrays, kth array denotes the spiketimes for the kth synapse, in chronological order and in [ms]
        tdur: duration of the simulation [ms]
        dt: timestep of the simulation [ms]
        conffile_name: name of the configuration file where the neuron parameters and the morphology are stated
        numsyn: number of synapses (should also be the length of the arrays [syndend], [synloc],...)
        syndend: array of integers, denoting the indices of the dendrites where a synapse is located
        synloc: array of floats between 0 and 1, denote the positions of the synapses in [syndend] on their respective dendrites
        gbar: array of weight [uS], choose within physiological range (~1e-3 uS)
        decay: array of decaytimes of the synapses [ms]
        E_rev: array of reversal potentials of the synapses [mV]
    -output:
        V_soma: voltage trace at soma
        time: corresponding time series
    '''
    q = greensFunctionCalculator(HH=0, conffilename = conffile_name)
    N = np.power(2,18)
    dtGreen = 0.01*1e-3
    Gt_synsyn = [] 
    Gt_synsoma = [] 
    GIt_synsoma = []
    GIt_awaysoma = []
    t = []
    for i in range(numsyn):
        Gt_synsyntemp, Gt_synsomatemp, GIt_synsomatemp, GIt_awaysomatemp, ttemp = q.greensFunctionTime(syndend[i],synloc[i], dt=dtGreen, N=N, tmax = 0.3)
        Gt_synsyn.append(Gt_synsyntemp)
        Gt_synsoma.append(Gt_synsomatemp)
        GIt_synsoma.append(GIt_synsomatemp)
        GIt_awaysoma.append(GIt_awaysomatemp)
        t.append(ttemp)
    neuron = pneuron(numsyn, np.array(Gt_synsoma), np.array(Gt_synsyn), np.array(t[0]), weights = gbar, E_r=E_rev, taus = decay, conffilename = conffile_name)
    V_soma, V_syns, time = neuron.run_syn(tdur, dt, spiketimes)
    
    return V_soma, time
    
    
if __name__=='__main__' :
    import matplotlib.pyplot as pl
    pl.figure()
    
    V_soma, time = compute_system([[20,20.5,21.,21.5,22,23,24],[50.,70]], 200, 0.025, numsyn=2, syndend = [0,1], synloc = [0.5,0.8], gbar=[0.0001,0.0004], decay = [5.,3.], E_rev = [-10.,-20.])
    pl.plot(time, V_soma,'b', label='Greens')
    
    import explicitIOD as ex
    Vm = ex.runModel_syn(tdur = 200, spiketimes = [[20,20.5,21.,21.5,22,23,24],[50.,70]], syndends = [0,1], synlocs = [0.5,0.8], synweights = [0.0001,0.0004], syntaus=[5.,3.], E_rs=[-10.,-20.])
    pl.plot(Vm['t'], Vm['vmsoma'],'r', label='NEURON')

    S =20
    pl.legend(loc=0)
    pl.xlabel('time (ms)',size=S)
    pl.ylabel('soma Vm (mV)',size=S)
    pl.xlim((0,155))
    pl.show()
