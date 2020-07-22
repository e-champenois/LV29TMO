import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd

def drawNormalizedSASE( energyGrid, photonEnergy, sigmaTrain, sigmaSpike, nSpikes ):
    spikeEnergies = np.random.normal(photonEnergy, sigmaTrain, nSpikes)[:,np.newaxis]
    SASEpulse = np.sum(np.exp( -(spikeEnergies - energyGrid)**2 / (2*sigmaSpike) ),axis=0)
    return (SASEpulse / np.sum(SASEpulse)).flatten()

def drawNumberOfPhotons( nPhAvg, percentVariability ):
    return np.random.normal(nPhAvg, 1e-2*percentVariability*nPhAvg, 1)[0]

def drawAugerEvent( nAuger ):
    return np.random.poisson(nAuger, 1)[0]

def makeSpooktroscopyMeasurement( energyGrid_eV, photonEnergy_eV, sigmaTrain_eV, sigmaSpike_eV, nSpikes,
                                  nPh, percentVariability,
                                  focusDiameter_cm,
                                  sigmaLinearInterpolation_cm2,
                                  moleculesInFocus):
    
    SASEpulse = drawNormalizedSASE( energyGrid_eV, photonEnergy_eV, sigmaTrain_eV, sigmaSpike_eV, nSpikes )
    photonsInPulse = drawNumberOfPhotons( nPh, percentVariability )
    calibratedPulse = photonsInPulse * SASEpulse
    
    augerAverage = np.sum(moleculesInFocus * calibratedPulse * sigmaLinearInterpolation_cm2(energyGrid_eV) / (np.pi*focusDiameter_cm**2/4.))
    augerMeasured = drawAugerEvent( augerAverage )
    
    return calibratedPulse, augerMeasured, photonsInPulse

def spooktroscopyExperiment( nMeasurements,
                             energyGrid_eV, photonEnergy_eV, sigmaTrain_eV, sigmaSpike_eV, nSpikes,
                             nPh, percentVariability,
                             focusDiameter_cm,
                             sigmaLinearInterpolation_cm2,
                             moleculesInFocus):
    
    pulses = []
    auger = []
    photons = []
    for idx in range(nMeasurements):
        calibratedPulse, augerMeasured, photonsInPulse = makeSpooktroscopyMeasurement( energyGrid_eV, photonEnergy_eV, sigmaTrain_eV, sigmaSpike_eV, nSpikes,
                                                                               nPh, percentVariability,
                                                                               focusDiameter_cm,
                                                                               sigmaLinearInterpolation_cm2,
                                                                               moleculesInFocus)
        auger.append(augerMeasured)
        photons.append(photonsInPulse)
        pulses.append(calibratedPulse)
            
    return np.array(pulses), np.array(auger), np.array(photons)