#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the Kramers-Kronig Calculator software package.
#
# Copyright (c) 2013 Benjamin Watts, Daniel J. Lauk
#
# The software is licensed under the terms of the zlib/libpng license.
# For details see LICENSE.txt

"""This module implements the Kramers-Kronig transformation."""

#import logging, sys
#logger = logging.getLogger(__name__)
#if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)
    #logging.StreamHandler(stream=sys.stdout)

import math
import numpy
#import os
from kkcalc import data
import specclass
import brennan


def calc_relativistic_correction(stoichiometry):
    """Calculate the relativistic correction to the Kramers-Kronig transform.

    Parameters:
    -----------
    stoichiometry : array of integer/float pairs
        Each pair in the list consists of an atomic number and the relative proportion of that element.

    Returns
    -------
    This function returns a ``float`` holding the relativistic
    corection to the Kramers-Kronig transform.

    """
    correction = 0
    for z, n in stoichiometry:
        #correction += (z - (z/82.5)**2.37) * n
        correction += (0 - (z/82.5)**2.37) * n
    return correction


def KK_General_PP(Eval_Energy, Energy, imaginary_spectrum, orders, relativistic_correction):
    """Calculate Kramers-Kronig transform with "Piecewise Polynomial"
    algorithm plus the Biggs and Lighthill extended data.

    Parameters
    ----------
    Eval_Energy : numpy vector of `float`
        Set of photon energies describing points at which to evaluate the real spectrum
    Energy : numpy vector of `float`
        Set of photon energies describing intervals for which each row of `imaginary_spectrum` is valid
    imaginary_spectrum : two-dimensional `numpy.array` of `float`
        The array consists of columns of polynomial coefficients belonging to the power terms indicated by 'order'
    orders : numpy vector of integers
        The vector represents the polynomial indices corresponding to the columns of imaginary_spectrum
    relativistic_correction : float
        The relativistic correction to the Kramers-Kronig transform.
        You can calculate the value using the `calc_relativistic_correction` function.

    Returns
    -------
    This function returns the real part of the scattering factors evaluated at photon energies specified by Eval_Energy.

    """
    #logger = logging.getLogger(__name__)
    print("Calculate Kramers-Kronig transform using general piecewise-polynomial algorithm")
    # Need to build x-E-n arrays
    X = numpy.tile(Energy[:,numpy.newaxis,numpy.newaxis],(1,len(Eval_Energy),len(orders)))
    E = numpy.tile(Eval_Energy[numpy.newaxis,:,numpy.newaxis],(len(Energy)-1,1,len(orders)))
    C = numpy.tile(imaginary_spectrum[:,numpy.newaxis,:],(1,len(Eval_Energy),1))
    N = numpy.tile(orders[numpy.newaxis,numpy.newaxis,:],(len(Energy)-1,len(Eval_Energy),1))
    poles = numpy.equal(X,numpy.tile(Eval_Energy[numpy.newaxis,:,numpy.newaxis],(len(Energy),1,len(orders))))
    
    # all N, ln(x+E) and ln(x-E) terms and poles
    Integral = numpy.sum(-C*(-E)**N*numpy.log(numpy.absolute((X[1:,:,:]+E)/(X[:-1,:,:]+E)))-C*E**N*(1-poles[1:,:,:])*numpy.log(numpy.absolute((X[1:,:,:]-E+poles[1:,:,:])/((1-poles[:-1,:,:])*X[:-1,:,:]+poles[:-1,:,:]*X[[0]+list(range(len(Energy)-2)),:,:]-E))),axis=(0,2))
    
    if numpy.any(orders<=-2): # N<=-2, ln(x) terms
        i = [slice(None,None,None),slice(None,None,None),orders<=-2]
        Integral += numpy.sum(C[i]*((-E[i])**N[i]+E[i]**N[i])*numpy.log(numpy.absolute((X[1:,:,orders<=-2])/(X[:-1,:,orders<=-2]))),axis=(0,2))
    
    if numpy.any(orders>=0): # N>=0,  x^k terms
        for ni in numpy.where(orders>=0)[0]:
            i = [slice(None,None,None),slice(None,None,None),ni]
            n = orders[ni]
            for k in range(n,0,-2):
                Integral += numpy.sum(C[i]/float(-k)*2*E[i]**(n-k)*(X[1:,:,ni]**k-X[:-1,:,ni]**k),axis=0)
    
    if numpy.any(orders <=-3): # N<=-3, x^k terms
        for ni in numpy.where(orders<=-3)[0]:
            i = [slice(None,None,None),slice(None,None,None),ni]
            n = orders[ni]
            for k in range(n+2,0,2):
                Integral += numpy.sum(C[i]/float(k)*((-1)**(n-k)+1)*E[i]**(n-k)*(X[1:,:,ni]**k-X[:-1,:,ni]**k),axis=0)
    
    print("Done!")
    return Integral / math.pi + relativistic_correction

def KK_PP(Eval_Energy, Energy, imaginary_spectrum, relativistic_correction):
    """Calculate Kramers-Kronig transform with "Piecewise Polynomial"
    algorithm plus the Biggs and Lighthill extended data.

    Parameters
    ----------
    Eval_Energy : numpy vector of `float`
        Set of photon energies describing points at which to evaluate the real spectrum
    Energy : numpy vector of `float`
        Set of photon energies describing intervals for which each row of `imaginary_spectrum` is valid
    imaginary_spectrum : two-dimensional `numpy.array` of `float`
        The array consists of five columns of polynomial coefficients: A_1, A_0, A_-1, A_-2, A_-3
    relativistic_correction : float
        The relativistic correction to the Kramers-Kronig transform.
        You can calculate the value using the `calc_relativistic_correction` function.

    Returns
    -------
    This function returns the real part of the scattering factors evaluated at photon energies specified by Eval_Energy.

    """
    #logger = logging.getLogger(__name__)
    print("Calculate Kramers-Kronig transform using (n from 1 to -3) piecewise-polynomial algorithm")
    X1 = Energy[0:-1]
    X2 = Energy[1:]
    E = numpy.tile(Eval_Energy, (len(Energy)-1, 1)).T
    Full_coeffs = imaginary_spectrum.T
    Symb_1 = (( Full_coeffs[0, :]*E+Full_coeffs[1, :])*(X2-X1)+0.5*Full_coeffs[0, :]*(X2**2-X1**2)-(Full_coeffs[3, :]/E+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute(X2/X1))+Full_coeffs[4, :]/E*(X2**-1-X1**-1))
    Symb_2 = ((-Full_coeffs[0, :]*E+Full_coeffs[1, :])*(X2-X1)+0.5*Full_coeffs[0, :]*(X2**2-X1**2)+(Full_coeffs[3, :]/E-Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute(X2/X1))-Full_coeffs[4, :]/E*(X2**-1-X1**-1))+(Full_coeffs[0, :]*E**2-Full_coeffs[1, :]*E+Full_coeffs[2, :]-Full_coeffs[3, :]*E**-1+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute((X2+E)/(X1+E)))
    Symb_3 = (1-1*((X2==E)|(X1==E)))*(Full_coeffs[0, :]*E**2+Full_coeffs[1, :]*E+Full_coeffs[2, :]+Full_coeffs[3, :]*E**-1+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute((X2-E+1*(X2==E))/(X1-E+1*(X1==E))))
    Symb_B = numpy.sum(Symb_2 - Symb_1 - Symb_3, axis=1)  # Sum areas for approximate integral
    # Patch singularities
    hits = Energy[1:-1]==E[:,0:-1]
    E_hits = numpy.append(numpy.insert(numpy.any(hits, axis=0),[0,0],False),[False,False])
    Eval_hits = numpy.any(hits, axis=1)
    X1 = Energy[E_hits[2:]]
    XE = Energy[E_hits[1:-1]]
    X2 = Energy[E_hits[:-2]]
    C1 = Full_coeffs[:, E_hits[2:-1]]
    C2 = Full_coeffs[:, E_hits[1:-2]]
    Symb_singularities = numpy.zeros(len(Eval_Energy))
    Symb_singularities[Eval_hits] = (C2[0, :]*XE**2+C2[1, :]*XE+C2[2, :]+C2[3, :]*XE**-1+C2[4, :]*XE**-2)*numpy.log(numpy.absolute((X2-XE)/(X1-XE)))
    # Finish things off
    KK_Re = (Symb_B-Symb_singularities) / (math.pi*Eval_Energy) + relativistic_correction
    print("Done!")
    return KK_Re

def improve_accuracy(Full_E, Real_Spectrum, Imaginary_Spectrum, relativistic_correction, tolerance, recursion=50):
    """Calculate extra data points so that a linear interpolation is more accurate.
    
    Parameters
    ----------
    Full_E : numpy vector of `float`
        Set of photon energies describing intervals for which each row of `imaginary_spectrum` is valid
    Real_Spectrum : numpy vector of `float`
        The real part of the spectrum corresponding to magnitudes at photon energies in Full_E
    Imaginary_Spectrum : two-dimensional `numpy.array` of `float`
        The array consists of five columns of polynomial coefficients: A_1, A_0, A_-1, A_-2, A_-3
    relativistic_correction : float
        The relativistic correction to the Kramers-Kronig transform.
        (You can calculate the value using the `calc_relativistic_correction` function.)
    tolerance : float
        Level of error in linear extrapolation of data values to be allowed.
    recursion : integer
        Number of times an energy interval can be halved before giving up.

    Returns
    -------
    This function returns a numpy array with three columns respectively representing photon energy, the real spectrum and the imaginary spectrum.
    """
    print("Improve data accuracy")
    new_points = numpy.cumsum(numpy.ones((len(Full_E)-2,1),dtype=numpy.int8))+1
    Im_values = data.coeffs_to_ASF(Full_E, numpy.vstack((Imaginary_Spectrum,Imaginary_Spectrum[-1])))
    #plot_Im_values = Im_values
    Re_values = Real_Spectrum
    E_values = Full_E
    temp_Im_spectrum = Imaginary_Spectrum[1:]
    count = 0
    improved = 1
    total_improved_points = 0
    while count<recursion and numpy.sum(improved)>0:
        #get E_midpoints
        midpoints = (E_values[new_points-1]+E_values[new_points])/2.
        #evaluate at new points
        Im_midpoints = data.coeffs_to_ASF(midpoints, temp_Im_spectrum)
        Re_midpoints = KK_PP(midpoints, Full_E, Imaginary_Spectrum, relativistic_correction)
        #evaluate error levels
        Im_error = abs((Im_values[new_points-1]+Im_values[new_points])/2. - Im_midpoints)
        Re_error = abs((Re_values[new_points-1]+Re_values[new_points])/2. - Re_midpoints)
        improved = (Im_error>tolerance) | (Re_error>tolerance)
        print(str(numpy.sum(improved))+" points (out of "+str(len(improved))+") can be improved in pass number "+str(count+1)+".")
        total_improved_points += numpy.sum(improved)
        #insert new points and values
        Im_values = numpy.insert(Im_values,new_points[improved],Im_midpoints[improved])
        Re_values = numpy.insert(Re_values,new_points[improved],Re_midpoints[improved])
        E_values = numpy.insert(E_values,new_points[improved],midpoints[improved])
        #prepare for next loop
        temp_Im_spectrum =numpy.repeat(temp_Im_spectrum[improved],2,axis=0)
        new_points = numpy.where(numpy.insert(numpy.zeros(Im_values.shape, dtype=numpy.bool),new_points[improved],True))[0]
        new_points = numpy.vstack((new_points, new_points+1)).T.flatten()
        count += 1

    print("Improved data accuracy by inserting "+str(total_improved_points)+" extra points.")
    return numpy.vstack((E_values,Re_values,Im_values)).T
    
def kk_calculate_real(NearEdgeDataFile, ChemicalFormula, load_options=None, input_data_type=None, merge_points=None, add_background=False, fix_distortions=False, curve_tolerance=None, curve_recursion=50):
    """Do all data loading and processing and then calculate the kramers-Kronig transform.
    Parameters
    ----------
    NearEdgeDataFile : string
        Path to file containg near-edge data
    ChemicalFormula : string
        A standard chemical formula string consisting of element symbols, numbers and parentheses.
    merge_points : list or tuple pair of `float` values, or None
        The photon energy values (low, high) at which the near-edge and scattering factor data values
        are set equal so as to ensure continuity of the merged data set.

    Returns
    -------
    This function returns a numpy array with columns consisting of the photon energy, the real and the imaginary parts of the scattering factors.
    """
    Stoichiometry = data.ParseChemicalFormula(ChemicalFormula)
    Relativistic_Correction = calc_relativistic_correction(Stoichiometry)
    Full_E, Imaginary_Spectrum = data.calculate_asf(Stoichiometry)
    if NearEdgeDataFile is not None:
        NearEdge_Data = data.convert_data(data.load_data(NearEdgeDataFile, load_options),FromType=input_data_type,ToType='asf')
        Full_E, Imaginary_Spectrum = data.merge_spectra(NearEdge_Data, Full_E, Imaginary_Spectrum, merge_points=merge_points, add_background=add_background, fix_distortions=fix_distortions)
    Real_Spectrum = KK_PP(Full_E, Full_E, Imaginary_Spectrum, Relativistic_Correction)
    if curve_tolerance is not None:
        output_data = improve_accuracy(Full_E,Real_Spectrum,Imaginary_Spectrum, Relativistic_Correction, curve_tolerance, curve_recursion)
    else:
        Imaginary_Spectrum_Values = data.coeffs_to_ASF(Full_E, numpy.vstack((Imaginary_Spectrum,Imaginary_Spectrum[-1])))
        output_data = numpy.vstack((Full_E,Real_Spectrum,Imaginary_Spectrum_Values)).T
    return output_data

if __name__ == '__main__':
    #use argparse here to get command line arguments
    #process arguments and pass to a pythonic function
    
    Stoichiometry = data.ParseChemicalFormula('NaUF5')
    #Stoichiometry = data.ParseChemicalFormula('GaAs')
    Relativistic_Correction = calc_relativistic_correction(Stoichiometry)
    ASF_E, ASF_Data = data.calculate_asf(Stoichiometry)
    ASF_E2 = numpy.empty(0,dtype=float)
    ASF_Data1 = numpy.zeros((0,5))
    from collections import deque
    temp = deque()
    for i,e in enumerate(ASF_E):
        if e<1000: continue
        else: 
            ASF_E2 = numpy.append(ASF_E2,e)
            if (i < len(ASF_Data)):
                temp.append(ASF_Data[i,:])
    ASF_Data1 = numpy.array(temp)
    ASF_Data3 = data.coeffs_to_linear(ASF_E2, ASF_Data1, 0.1)
    ASF_Data2 = data.coeffs_to_ASF(ASF_E2, numpy.vstack((ASF_Data1,ASF_Data1[-1])))
    Re_data = KK_PP(ASF_E2, ASF_E2, ASF_Data1, Relativistic_Correction)

    # Get splice points
    raw_file = data.load_data_with_I0("NaUF5.spec",load_options={'data_column': 4, 'I0_column': 10})
    splice_eV = numpy.array([raw_file[0,0], raw_file[-1,0]])  # data limits
    Full_E, Imaginary_Spectrum, NearEdgeData, splice_ind  = data.merge_spectra(raw_file, 
                                                                               ASF_E, 
                                                                               ASF_Data, 
                                                                               merge_points=splice_eV, 
                                                                               add_background=False,
                                                                               fix_distortions=True,
                                                                               plotting_extras=True)
    
    KK_Real_Spectrum = KK_PP(Full_E, Full_E, Imaginary_Spectrum, Relativistic_Correction)
    
    print("Loading Brennan & Cowan Table",end="  ",flush=True)
    br = brennan.brennan()
    print("..done")
    import pylab
    
    pylab.figure()
    print("Calculating brennan & Cowan values",end="  ",flush=True)
    br_e = numpy.linspace(NearEdgeData[0,0],NearEdgeData[-1,0],1000)
    br_fp = numpy.zeros_like(br_e)
    br_fdp = numpy.zeros_like(br_e)
    for i,e in enumerate(br_e):
        for Z,n in Stoichiometry:
            fp, fdp = br.at_energy(e/1000, data.LIST_OF_ELEMENTS[Z-1])
            br_fp[i] += n*fp
            br_fdp[i] += n*fdp
    print("..done")
    
    pylab.plot(ASF_Data3[0],ASF_Data3[1],':r')
    pylab.plot(ASF_E2,Re_data,':b')
    pylab.plot(br_e,br_fdp,'-r')
    pylab.plot(br_e,br_fp,'-b')

    pylab.plot(NearEdgeData[:,0],NearEdgeData[:,1],'+c')
    pylab.plot(Full_E,KK_Real_Spectrum,'--g')
    Es = [17100,17150,17160,17166,17180,17200,17220,17300]
    fps = [-13.6595,-16.2754,-17.5141,-19.1306,-16.6550,-12.8490,-13.8983,-11.7568]
    fdps = [6.6971, 5.7636, 6.0041, 6.5918, 17.0275, 11.0402, 11.5513, 11.0343]
    
    pylab.plot(Es,fps,'ok')
    pylab.plot(Es,fdps,'om')
    
    #pylab.plot(ASF_Data3[0],ASF_Data3[1],'r-')
    #pylab.xscale('log')
    pylab.xlim(raw_file[0,0], raw_file[-1,0])
    pylab.ylim(-35,20)
    pylab.show()
