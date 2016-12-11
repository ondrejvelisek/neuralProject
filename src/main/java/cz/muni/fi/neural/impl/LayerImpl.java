package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.Utils;
import cz.muni.fi.neural.lib.ActivationFunction;
import cz.muni.fi.neural.lib.Layer;
import cz.muni.fi.neural.lib.Neuron;
import cz.muni.fi.neural.lib.WeightsInitAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class LayerImpl implements Layer {

	private List<Neuron> neurons;

	public LayerImpl(int previousSize, int size, ActivationFunction ac, WeightsInitAlgorithm wia) {
		if (previousSize <= 0 || size <= 0) {
			throw new IllegalArgumentException("Number of neurons has to be greater than 0");
		}

		this.neurons = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			this.neurons.add(new NeuronImpl(previousSize+1, ac, wia));
		}

	}

	public List<Double> computeOutput(List<Double> inputs) {
		Utils.checkEqualSize(getInputSize(), inputs);

		List<Double> inputWithBias = new ArrayList<>(inputs);
		inputWithBias.add(1.0);

		List<Double> neuronOutputs = new ArrayList<>();
		for (Neuron neuron : neurons) {
			neuronOutputs.add(neuron.computeOutput(inputWithBias));
		}
		return neuronOutputs;
	}

	public int getInputSize() {

		return neurons.get(0).getInputSize()-1;

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

	@Override
	public String toString() {
		return "\nLayer{" +
				"\nneurons=" + neurons +
				"\n}";
	}
}