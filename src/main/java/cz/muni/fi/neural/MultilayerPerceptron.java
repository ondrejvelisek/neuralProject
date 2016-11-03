package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
			neurons.add(new TanhNeuron(inputSize));
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

	public void learn(Map<List<Double>, List<Double>> trainingSet) {

		List<List<List<Double>>> newWeights = new ArrayList<>();

		for (int i = 0; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			newWeights.add(new ArrayList<>());

			for (int n = 0; n < layer.getNeurons().size(); n++) {
				Neuron neuron = layer.getNeurons().get(n);
				newWeights.get(i).add(new ArrayList<>());

				for (int j = 0; j < neuron.getWeights().size(); j++) {
					double weight = neuron.getWeights().get(j);

					double gradientEwji = 0;
					for (Map.Entry<List<Double>, List<Double>> sample: trainingSet.entrySet()) {
						List<Double> sampleInput = sample.getKey();
						List<Double> desireOutput = sample.getValue();

						// we obtain last results in all neurons
						compute(sampleInput);
						double oj = neuron.getLastDerivatedOutput();
						double yi;
						if (i == 0) {
							yi = sampleInput.get(j);
						} else {
							Neuron connectedNeuron = layers.get(i - 1).getNeurons().get(j);
							yi = connectedNeuron.getLastOutput();
						}

						double gradientEkyj = 0;
						if (i == layers.size() - 1) {
							gradientEkyj = neuron.getLastOutput() - desireOutput.get(n);
							neuron.setGradient(gradientEkyj);
						} else {
							Layer above = layers.get(i + 1);
							gradientEkyj = 0;
							for (Neuron r : above.getNeurons()) {
								gradientEkyj += r.getGradient() * r.getLastDerivatedOutput() * r.getWeights().get(n);
							}
						}

						double gradientEkwji = gradientEkyj * oj * yi;

						gradientEwji += gradientEkwji;
					}

					double deltaW = -0.01 * gradientEwji;

					newWeights.get(i).get(n).add(weight + deltaW);

				}

			}

		}

		for (int i = 0; i < layers.size(); i++) {
			Layer layer = layers.get(i);

			for (int n = 0; n < layer.getNeurons().size(); n++) {
				Neuron neuron = layer.getNeurons().get(n);

				for (int j = 0; j < neuron.getWeights().size(); j++) {

					neuron.getWeights().set(j, newWeights.get(i).get(n).get(j));

				}
			}
		}


	}

	public int getInputSize() {
		return layers.get(0).getInputSize();
	}

	public int getOutputSize() {
		return layers.get(layers.size() - 1).getOutputSize();
	}

}
