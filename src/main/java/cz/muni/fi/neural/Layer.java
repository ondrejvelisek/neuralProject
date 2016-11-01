package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class Layer implements NeuralNetwork {

	private List<Neuron> neurons;

	public Layer(List<Neuron> neurons) {
		if (neurons.isEmpty()) {
			throw new IllegalArgumentException("Neurons are empty");
		}
		int previousSize = neurons.get(0).getInputSize();
		for (Neuron neuron : neurons) {
			if (neuron.getInputSize() != previousSize) {
				throw new IllegalArgumentException("All neurons in one layer has to have same input size");
			}
			previousSize = neuron.getInputSize();
		}

		this.neurons = neurons;
	}

	public List<Double> compute(List<Double> inputs) {
		if (inputs.size() != getInputSize()) {
			throw new IllegalArgumentException("Cannot compute different size of input");
		}

		List<Double> result = new ArrayList<>();
		for (Neuron neuron : neurons) {
			result.add(neuron.compute(inputs).get(0));
		}
		return result;

	}

	public int getInputSize() {

		return neurons.get(0).getInputSize();

	}

	public int getOutputSize() {
		return neurons.size();
	}


}
