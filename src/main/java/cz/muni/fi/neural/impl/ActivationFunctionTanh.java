package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.lib.ActivationFunction;

/**
 * Created by Simon on 06.11.2016.
 */
public class ActivationFunctionTanh implements ActivationFunction {

    private double steepness;

    public ActivationFunctionTanh(double steepness) {
        assert(steepness > 0);
        this.steepness = 1;
    }

    public double computeOutput(double innerPotential){
        return Math.tanh(steepness * innerPotential);
    }

    public double derivationOutput(double output){
        return steepness * (1 - output)*(1 + output);
    }

}
