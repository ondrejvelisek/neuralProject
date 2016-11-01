package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class BiasNeuron extends Neuron {

	private List<Double> weights;
	private Double bias;

	public BiasNeuron(int inputSize, Double bias) {
		this.weights = generateWeights(inputSize);
		this.bias = bias;
	}

	private List<Double> generateWeights(int inputSize) {
		Random r = new Random();
		List<Double> list = new ArrayList<>();
		for (int i = 0; i < inputSize; i++) {
			list.add(-1 + 2 * r.nextDouble());
		}
		return list;

	}

	public List<Double> compute(List<Double> inputs) {
		if (inputs.size() != weights.size()) {
			throw new IllegalStateException();
		}

		if (getInnerPotential(inputs) > bias) {
			return Collections.singletonList(1.0);
		} else {
			return Collections.singletonList(0.0);
		}

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

}
