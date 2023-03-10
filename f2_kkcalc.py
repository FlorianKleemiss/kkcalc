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
    Symb_1 = (( Full_coeffs[0, :]*E + Full_coeffs[1, :])*(X2-X1) + 0.5*Full_coeffs[0, :]*(X2**2-X1**2) - (Full_coeffs[3, :]/E+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute(X2/X1)) + Full_coeffs[4, :]/E*(X2**-1-X1**-1))
    Symb_2 = ((-Full_coeffs[0, :]*E + Full_coeffs[1, :])*(X2-X1) + 0.5*Full_coeffs[0, :]*(X2**2-X1**2) + (Full_coeffs[3, :]/E-Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute(X2/X1)) - Full_coeffs[4, :]/E*(X2**-1-X1**-1)) + (Full_coeffs[0, :]*E**2-Full_coeffs[1, :]*E+Full_coeffs[2, :]-Full_coeffs[3, :]*E**-1+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute((X2+E)/(X1+E)))
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
    #KK_Re = (Symb_B) / (math.pi*Eval_Energy) + relativistic_correction
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

    from matplotlib import pyplot as plt
    fig = plt.figure()
    pylab = fig.add_subplot(1,1,1)
    
    #HL19
    #Es = [17100,17150,17181,17203,17220,17250,17300]
    #fps = numpy.array([-14.3032, -16.7177,-16.1259,-13.9231,-14.0630,-13.4530,-12.6767])
    #fdps = numpy.array([6.8452, 6.4845, 17.2928, 10.0615, 11.8307, 11.1966, 11.8952])
    ##fps *= 2.0
    ##fdps *= 2.0
    #Stoichiometry = data.ParseChemicalFormula('U')
    #spec_test = specclass.Spec("specs/HL19-NaU2F9-manual.spec")
    #pylab.set_title("HL19 NaU2F9")
    

    #HL14
    Es = [17100,17150,17160,17166,17180,17200,17220,17300]
    fps = [-13.6595,-16.2754,-17.5141,-19.1306,-16.6550,-12.8490,-13.8983,-11.7568]
    fdps = [6.6971, 5.7636, 6.0041, 6.5918, 17.0275, 11.0402, 11.5513, 11.0343]
    Stoichiometry = data.ParseChemicalFormula('U')
    spec_test = specclass.Spec("specs/HL14-NaUF5-manual.spec")
    pylab.set_title("HL14 NaUF5")

    #HL17
    #temp = numpy.array([[17000, -10.9052, 4.7638],
    #[17150, -14.9665, 4.4791 ],
    #[17176, -19.1501, 8.1872 ],
    #[17184, -15.7981, 12.4364],
    #[17200, -11.6748, 10.3490],
    #[17210, -12.6413, 9.4839 ],
    #[17250, -11.5393, 9.7332 ]]).T
    #Es = temp[0].tolist()
    #fps = temp[1].tolist()
    #fdps =  temp[2].tolist()
    ##
    ###resolution limit 0.57
    ###temp_low_res = numpy.array([[17000, -10.8578, 5.1398 ],
    ###[17150, -14.9456, 3.9882 ],
    ###[17176, -19.5159, 8.3557 ], 
    ###[17184, -16.2605, 12.4530],
    ###[17200, -12.0690, 10.4034],
    ###[17210, -12.6650, 9.6555 ],
    ###[17250, -11.8404, 9.8187 ]]).T
    ###Es_low_res = temp_low_res[0].tolist()
    ###fps_low_res = temp_low_res[1].tolist()
    ###fdps_low_res =  temp_low_res[2].tolist()
    #Stoichiometry = data.ParseChemicalFormula('U')
    #spec_test = specclass.Spec("specs/HL1-Cs2UO2TiO4-manual.spec")
    #pylab.set_title("HL17 Cs2UO2TiO4")

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
    #ASF_Data1 = numpy.array(temp)
    #ASF_Data3 = data.coeffs_to_linear(ASF_E2, ASF_Data1, 0.1)
    #ASF_Data2 = data.coeffs_to_ASF(ASF_E2, numpy.vstack((ASF_Data1,ASF_Data1[-1])))
    #Re_data = KK_PP(ASF_E2, ASF_E2, ASF_Data1, Relativistic_Correction)

    # Get spec points
    spec_test.evaluate()
    raw_speccy = spec_test.output_E_a_array(a="sca")
    splice_eV = numpy.array([raw_speccy[0,0], raw_speccy[-1,0]])  # data limits
    Full_E, Imaginary_Spectrum, NearEdgeData, splice_ind, p1  = data.merge_spectra(raw_speccy, 
                                                                               ASF_E, 
                                                                               ASF_Data, 
                                                                               add_background=True, #abused now for sclaing last and first 3 values
                                                                               fix_distortions=True,
                                                                               plotting_extras=True)
    
    print("Loading Brennan & Cowan Table",end="  ",flush=True)
    br = brennan.brennan()
    print("..done")
    
    Full_E2, Full_Coefs_br, NearEdgeData2 = data.merge_spectra_brennan(raw_speccy, 
                                                                               ASF_E, 
                                                                               br,
                                                                               92,
                                                                               add_background=True, #abused now for sclaing last and first 3 values
                                                                               fix_distortions=True)
    
    KK_Real_Spectrum = KK_PP(Full_E, Full_E, Imaginary_Spectrum, Relativistic_Correction)

    #KK_Real_brennan = KK_PP(Full_E2, Full_E2, Full_Coefs_br, Relativistic_Correction)
    #null = 0

    def residual (par, spectrum, fdp):
        return numpy.sum(numpy.square((spectrum*par[0]+par[1]) - fdp))
    def jac(par,spectrum,fdp):
        return [sum(2*spectrum*spectrum*par[0]+2*spectrum*par[1]-2*spectrum*fdp),sum(2*spectrum*par[1]+2*par[1]-2*fdp)]
    
    import scipy
    x0 = numpy.array([4.0,3.0])
    i_start = 0
    dist = 2E20
    for i,e in enumerate(NearEdgeData[:,0]):
        dist_new = abs(e-min(Es))
        if dist_new < dist:
            dist = dist_new
        else:
            if i > 0:
                i_start = i -1
            else:
                i_start = 0
            break
    i_end = i_start
    dist = 2E20
    for i,e in enumerate(NearEdgeData[i_start:,0]):
        dist_new = abs(e-max(Es))
        if dist_new < dist:
            dist = dist_new
        else:
            if i+1 < len(NearEdgeData[i_start:,0]):
                i_end = i+1
            else:
                i_end = i
            break

    spectrum_part = NearEdgeData[i_start:i_end,1]
    energies_part = NearEdgeData[i_start:i_end,0]
    spec_fdps = []
    spec_es = []
    ref_fdps = fdps.copy()
    for E in Es:
        dist = 2E20
        found = False
        for i,e in enumerate(energies_part):
            dist_new = abs(e-E)
            if dist_new < dist:
                dist = dist_new
            else:
                spec_fdps.append(spectrum_part[i])
                spec_es.append(energies_part[i])
                found = True
                break
        if found == False:
            ref_fdps.pop(Es.index(E))
    spec_fdps = numpy.array(spec_fdps)
    results = scipy.optimize.minimize(residual, 
                                      x0, args=(spec_fdps, numpy.array(ref_fdps)),
                                      jac = jac,
                                      bounds = [(0.0,3.0),(-100,100)],
                                      options={'disp': True, 'gtol':1E-10, 'maxiter': 100})
    p = results.x
    figgy = plt.figure()
    axy = figgy.add_subplot(1,1,1)
    axy.scatter(spec_fdps,ref_fdps)
    x = numpy.linspace(min(fdps)-2,max(fdps)+1,200)
    axy.plot(x, x*p[0]+p[1])
    axy.plot(x, x)
    #plt.show()
    
    print("Calculating brennan & Cowan values",end="  ",flush=True)
    br_e = numpy.linspace(NearEdgeData[0,0],NearEdgeData[-1,0],2000)
    br_fp = numpy.zeros_like(br_e)
    br_fdp = numpy.zeros_like(br_e)
    for i,e in enumerate(br_e):
        for Z,n in Stoichiometry:
            fp, fdp = br.at_energy(e/1000, data.LIST_OF_ELEMENTS[Z-1])
            br_fp[i] += n*fp
            br_fdp[i] += n*fdp
    print("..done")
    
    #pylab.plot(ASF_Data3[0],ASF_Data3[1],':r')
    #pylab.plot(ASF_E2,Re_data,':b')
    pylab.plot(br_e,br_fdp,'-r')
    pylab.plot(br_e,br_fp,'-b')

    #corr = br_fp[0] + \
    #       ((NearEdgeData[:,1]-p1*NearEdgeData[:,0])-(NearEdgeData[0,1]-p1*NearEdgeData[0,0])) * \
    #       (br_fdp[-1]-br_fdp[0]) / \
    #       ((NearEdgeData[-1,1]-p1*NearEdgeData[-1,0])-(NearEdgeData[0,1]-p1*NearEdgeData[0,0]))
    #pylab.plot(NearEdgeData[:,0],corr,"--k")



    pylab.plot(NearEdgeData[:,0],NearEdgeData[:,1],'+c')
    pylab.plot(NearEdgeData[:,0],NearEdgeData[:,1]*p[0]+p[1],'+k')
    pylab.plot(Full_E,KK_Real_Spectrum,'--g')
    
    pylab.plot(Es,fps,'ok')
    pylab.plot(Es,fdps,'om')
    if 'fps_low_res' in locals():
        pylab.plot(Es_low_res,fps_low_res,'oy')
        pylab.plot(Es_low_res,fdps_low_res,'og')
    
    #pylab.plot(ASF_Data3[0],ASF_Data3[1],'r-')
    #pylab.xscale('log')
    pylab.set_xlim(raw_speccy[0,0], raw_speccy[-1,0])
    ys = numpy.concatenate((NearEdgeData[:,1],KK_Real_Spectrum[numpy.argmax(Full_E>raw_speccy[0,0]):],fdps,fps))
    y_min = numpy.min(ys)
    if y_min < 0:
        y_min *= 1.1
    else:
        y_min *= 0.9
    y_max = numpy.max(ys)
    if y_max < 0:
        y_max *= 0.9
    else:
        y_max *= 1.1
    pylab.set_ylim(y_min,y_max)
    pylab.set_xlabel("energy / eV")
    pylab.set_ylabel("scattering factor /e")
    plt.show()
    
