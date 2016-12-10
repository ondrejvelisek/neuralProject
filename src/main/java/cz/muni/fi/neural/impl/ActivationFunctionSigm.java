package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.lib.ActivationFunction;

/**
 * Created by Simon on 06.11.2016.
 */
public class ActivationFunctionSigm implements ActivationFunction {

	private double steepness;

	public ActivationFunctionSigm(double steepness) {
		assert(steepness > 0);
		this.steepness = steepness;
	}

	public double computeOutput(double innerPotential){
		return 1.0 / (1 + Math.exp(-steepness * innerPotential));
	}

	public double derivationOutput(double output){
		return steepness * output * (1 - output);
	}

}
