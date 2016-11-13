package cz.muni.fi.neural.algorithms;

import cz.muni.fi.neural.algorithms.ActivationFunction;

/**
 * Created by Simon on 06.11.2016.
 */
public class ActivationFunctionTanh implements ActivationFunction {

    public double computeOutput(double innerPotential){
       return Math.tanh(innerPotential);
    }

    public double derivationOutput(double y){
        return (1 - y)*(1 + y);
    }
}
