import numpy as np
import matplotlib.pyplot as plt

N = 8
O1 = 4
dVsLambda = 0.5
k = 2
n = np.arange(0,N,1)
nHalf = np.arange(0,N/2,1)
fig = plt.figure(figsize=(15,15))

########################################
#  把 beam vector 分成上下两部分，第二部分在第一部分的基础上乘以一个相位偏转
#  PI/4 * p , p = 0,1,2,3
########################################
for p in range(0,4):
    phaseRotateVectorFirstHalf = np.exp(1j*4*k*np.pi/(N*O1)*nHalf)
    phaseRotateVectorSecondHalf = phaseRotateVectorFirstHalf * np.exp(1j*p*np.pi/4)
    phaseRotateVector =np.concatenate((phaseRotateVectorFirstHalf,phaseRotateVectorSecondHalf), axis=0) 
    #print(phaseRotateVector)
    thetaAll = np.arange(0,2*np.pi,0.01)
    beamAmp = []
    for theta in thetaAll:
        ePhi = np.exp(1j*2*np.pi*dVsLambda*np.cos(theta)*n)
        BeamAmpForOneTheta = np.dot(phaseRotateVector,ePhi)
        beamAmp.append(BeamAmpForOneTheta)
    beamAmp = np.array(beamAmp)/N

    ax = plt.subplot(111, polar=True)
    ax.plot(thetaAll, abs(beamAmp),"--")
    #ax.vlines(beamAngle,0,16,'r--')
    ax.grid(True)
    ax.set_theta_offset(-1*np.pi/2)
    
########################################
#  把 beam vector 分成上下两部分，只画上半部分的
########################################
phaseRotateVector2 = np.exp(1j*4*k*np.pi/(N*O1)*nHalf)
thetaAll = np.arange(0,2*np.pi,0.01)
#print(phaseRotateVector2)
beamAmp = []
for theta in thetaAll:
    ePhi = np.exp(1j*2*np.pi*dVsLambda*np.cos(theta)*nHalf)
    BeamAmpForOneTheta = np.dot(phaseRotateVector2,ePhi)
    beamAmp.append(BeamAmpForOneTheta)
beamAmp = np.array(beamAmp)/(N/2)
ax = plt.subplot(111, polar=True)
ax.plot(thetaAll, abs(beamAmp),"-")
ax.grid(True)
ax.set_theta_offset(-1*np.pi/2)
    
########################################
#  把 beam 用完整的，但是相位旋转的依然是 k
########################################
phaseRotateVector2 = np.exp(1j*4*k*np.pi/(N*O1)*n)
thetaAll = np.arange(0,2*np.pi,0.01)
beamAmp = []
for theta in thetaAll:
    ePhi = np.exp(1j*2*np.pi*dVsLambda*np.cos(theta)*n)
    BeamAmpForOneTheta = np.dot(phaseRotateVector2,ePhi)
    beamAmp.append(BeamAmpForOneTheta)
beamAmp = np.array(beamAmp)/N
ax = plt.subplot(111, polar=True)
ax.plot(thetaAll, abs(beamAmp),".")
ax.grid(True)
ax.set_theta_offset(-1*np.pi/2)

ax.legend(("0*pi/4","1*pi/4","2*pi/4","3*pi/4","N/2 normal","N normal"))   
