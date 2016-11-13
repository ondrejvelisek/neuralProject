package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class LayerImpl implements Layer {

	private List<Neuron> neurons;


	public LayerImpl(List<Neuron> neurons) {
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

	public List<Double> computeOutput(List<Double> inputs) {
		if (inputs.size() < getInputSize()) {
			throw new IllegalArgumentException("Cannot computeOutput different size of input");
		}

		List<Double> neuronOutputs = new ArrayList<>();
		for (Neuron neuron : neurons) {
			neuronOutputs.add(neuron.computeOutput(inputs));
		}

		return neuronOutputs;
	}

	public int getInputSize() {

		return neurons.get(0).getInputSize();

	}

	public void learn(Map<List<Double>, List<Double>> trainingSet) {
		throw new UnsupportedOperationException();
	}

	public int getOutputSize() {
		return neurons.size();
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	@Override
	public int getNeuronPosition(Neuron neuron) {
		return neurons.indexOf(neuron);
	}
}
