package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class Neuron {

	private List<Double> weights;
	private double lastOutput;
	private double lastDerivatedOutput;
	private double gradient;
	private ActivationFunction activationFunction;

	public Neuron(int inputSize) {
		this.weights = generateWeights(inputSize);
		activationFunction = new ActivationFunctionTanh();
	}

	private List<Double> generateWeights(int inputSize) {
		Random r = new Random();
		List<Double> list = new ArrayList<>();
		for (int i = 0; i < inputSize; i++) {
			list.add(-0.5 + 1 * r.nextDouble());
		}
		return list;

	}

	public List<Double> computeOutput(List<Double> inputs) {
		if (inputs.size() != weights.size()) {
			throw new IllegalStateException();
		}
		double innerPotential = computeInnerPotential(inputs);
		lastOutput = activationFunction.computeOutput(innerPotential);
		getDerivatedOutput(lastOutput); //?????
		return Collections.singletonList(lastOutput);
	}

	public void learn(Map<List<Double>, List<Double>> trainingSet) {
		throw new UnsupportedOperationException();
	}

	private double computeInnerPotential(List<Double> inputs) {

		double sum = 0;
		for(int i = 0; i < inputs.size(); i++) {
			sum += inputs.get(i) * weights.get(i);
		}
		return sum;

	}

	public int getInputSize() {
		return weights.size();
	}

	public double getLastOutput() {
		return lastOutput;
	}

	public double getLastDerivatedOutput() {
		return lastDerivatedOutput;
	}

	private double getDerivatedOutput(double y) {
		lastDerivatedOutput = (1 - y)*(1 + y);
		return lastDerivatedOutput;
	}

	public List<Double> getWeights() {
		return weights;
	}

	public double getGradient() {
		return gradient;
	}

	public void setGradient(double gradient) {
		this.gradient = gradient;
	}
}
