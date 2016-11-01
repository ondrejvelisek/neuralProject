package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class MultilayerPerceptron implements NeuralNetwork {

	private List<Layer> layers;

	public MultilayerPerceptron(int... neuronsInLayer) {
		if (neuronsInLayer.length <= 1) {
			throw new IllegalArgumentException("At least one layers have to be present in MPL. It means two numbers has to be provided.");
		}

		layers = new ArrayList<>();
		for (int i = 1; i < neuronsInLayer.length; i++) {
			layers.add(constructLayer(neuronsInLayer[i-1], neuronsInLayer[i]));
		}

	}

	private Layer constructLayer(int inputSize, int layerSize) {

		List<Neuron> neurons = new ArrayList<>();
		for (int i = 0; i < layerSize; i++) {
			neurons.add(new BiasNeuron(inputSize, 0.5));
		}
		return new Layer(neurons);
	}

	public List<Double> compute(List<Double> inputs) {
		if (inputs.size() != getInputSize()) {
			throw new IllegalArgumentException("inputs have to match inputSize of the network");
		}

		// Do not want to modify parameter of the method
		List<Double> currentResult = new ArrayList<>(inputs);
		for (Layer layer : layers) {
			currentResult = layer.compute(currentResult);
		}
		return currentResult;

	}

	public int getInputSize() {
		return layers.get(0).getInputSize();
	}

	public int getOutputSize() {
		return layers.get(layers.size() - 1).getOutputSize();
	}
}
