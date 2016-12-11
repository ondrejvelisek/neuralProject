package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.ConfigReader;
import cz.muni.fi.neural.Utils;
import cz.muni.fi.neural.lib.Neuron;
import cz.muni.fi.neural.lib.ActivationFunction;
import cz.muni.fi.neural.lib.Weight;
import cz.muni.fi.neural.lib.WeightsInitAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class NeuronImpl implements Neuron {
	public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME); //negetovat ho vsade

	private List<Weight> weights;
	private ActivationFunction activationFunction;

	public NeuronImpl(int inputSize, ActivationFunction af, WeightsInitAlgorithm wia) {
		this.weights = wia.initWeights(inputSize);
		this.activationFunction = af;
	}

	public double computeOutput(List<Double> inputs) {
		Utils.checkEqualSize(this.weights, inputs);

		double innerPotential = computeInnerPotential(inputs);
		return activationFunction.computeOutput(innerPotential);
	}

	private double computeInnerPotential(List<Double> inputs) {

		double sum = 0;
		for(int i = 0; i < weights.size(); i++) {
			if(ConfigReader.getInstance().neuronInputsDebug()){
				logger.info(""+inputs.get(i));
			}
			sum += inputs.get(i) * weights.get(i).getValue();
		}

		if(ConfigReader.getInstance().neuronInputsDebug()){
			logger.info("--------------------------");
		}

		return sum;

	}

	public int getInputSize() {
		if(weights == null) {
			return 0;
		}

		return weights.size();
	}

	public double derivationOutput(double output) {
		return activationFunction.derivationOutput(output);
	}

	public List<Weight> getWeights() {
		return weights;
	}

	@Override
	public List<Double> getWeightValues() {
		return weights.stream().map(Weight::getValue).collect(Collectors.toList());
	}

	@Override
	public void updateWeights(List<Double> weights) {
		Utils.checkEqualSize(this.weights, weights);

		IntStream.range(0, this.weights.size())
				.forEach(i -> this.weights.get(i).setValue(weights.get(i)));
	}

	@Override
	public String toString() {
		return "\n\tNeuron{" +
				"weights=" + weights +
				"}";
	}
}
