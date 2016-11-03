package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class TanhNeuron extends Neuron {

	private List<Double> weights;
	private double lastOutput;
	private double lastDerivatedOutput;
	private double gradient;

	public TanhNeuron(int inputSize) {
		this.weights = generateWeights(inputSize);
	}

	private List<Double> generateWeights(int inputSize) {
		Random r = new Random();
		List<Double> list = new ArrayList<>();
		for (int i = 0; i < inputSize; i++) {
			list.add(-0.5 + 1 * r.nextDouble());
		}
		return list;

	}

	public List<Double> compute(List<Double> inputs) {
		if (inputs.size() != weights.size()) {
			throw new IllegalStateException();
		}
		double pot = getInnerPotential(inputs);
		lastOutput = Math.tanh(pot);
		getDerivatedOutput(lastOutput);
		return Collections.singletonList(lastOutput);
	}

	@Override
	public void learn(Map<List<Double>, List<Double>> trainingSet) {
		throw new UnsupportedOperationException();
	}

	private double getInnerPotential(List<Double> inputs) {

		double sum = 0;
		for(int i = 0; i < inputs.size(); i++) {
			sum += inputs.get(i) * weights.get(i);
		}
		return sum;

	}

	public int getInputSize() {
		return weights.size();
	}

	@Override
	public double getLastOutput() {
		return lastOutput;
	}

	@Override
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

	@Override
	public double getGradient() {
		return gradient;
	}

	@Override
	public void setGradient(double gradient) {
		this.gradient = gradient;
	}
}
