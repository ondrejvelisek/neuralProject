package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.ConfigReader;
import cz.muni.fi.neural.Utils;
import cz.muni.fi.neural.lib.ActivationFunction;
import cz.muni.fi.neural.lib.Neuron;
import cz.muni.fi.neural.lib.Weight;
import cz.muni.fi.neural.lib.WeightsInitAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class NeuronBias implements Neuron {
	public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME); //negetovat ho vsade

	public double computeOutput(List<Double> inputs) {
		return 1;
	}

	public int getInputSize() {
		return 0;
	}

	public double derivationOutput(double output) {
		return 0;
	}

	public List<Weight> getWeights() {
		return new ArrayList<>();
	}

	@Override
	public List<Double> getWeightValues() {
		return new ArrayList<>();
	}

	@Override
	public void updateWeights(List<Double> weights) {
	}

}
