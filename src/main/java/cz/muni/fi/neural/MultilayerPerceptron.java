package cz.muni.fi.neural;

import java.util.*;
import java.util.logging.Logger;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class MultilayerPerceptron implements NeuralNetwork {
	public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

	private List<Layer> layers;

	public MultilayerPerceptron(List<Integer> layersStructure) {
		int numOfLayers = layersStructure.size();
		if (numOfLayers < 3) {
			throw new IllegalArgumentException("At least one hidden layer have to be present in MPL.");
		}

		layers = new ArrayList<>();
		for (int i = 1; i < numOfLayers; i++) {
			layers.add(constructLayer(layersStructure.get(i-1), layersStructure.get(i)));
		}

		if(ConfigReader.getInstance().initializationDebug()){
			logger.info("--------------------");
			logger.info("Multilayer perceptron builded: ");
			logger.info("Structure of MLP without input layer: ");
			for (int i = 0; i < layers.size(); i++) {
				logger.info(""+layers.get(i).getNeurons().size());
			}
			logger.info("--------------------");
		}

	}

	private Layer constructLayer(int prevLayerSize, int layerSize) {

		List<Neuron> neurons = new ArrayList<>();
		for (int i = 0; i < layerSize; i++) {
			neurons.add(new NeuronImpl(prevLayerSize));
		}
		return new LayerImpl(neurons);
	}

	public List<Double> computeOutput(List<Double> inputs) {
		if (inputs.size() < getInputSize()) {
			throw new IllegalArgumentException("inputs have to match inputSize of the network");
		}

		List<Double> inputsForNextLayer = new ArrayList<>(inputs);
		for (Layer currentLayer : layers) {
			inputsForNextLayer = currentLayer.computeOutput(inputsForNextLayer);
		}
		return inputsForNextLayer;
	}

	public void learn(Map<List<Double>, List<Double>> trainingSet) {

		int miniBatchSize = 10;
		Map<List<Double>, List<Double>> miniBatch = new HashMap<>();
		int t = 0;
		for (Map.Entry<List<Double>, List<Double>> trainingSample : trainingSet.entrySet()) {
			miniBatch.put(trainingSample.getKey(), trainingSample.getValue());
			if (miniBatch.size() == miniBatchSize) {

				miniLearn(t, miniBatch);

				miniBatch.clear();
				t++;
			}
		}
		miniLearn(t, miniBatch);
	}

	private void miniLearn(int time, Map<List<Double>, List<Double>> miniBatch) {

		Map<Neuron, List<Double>> miniGradient = computeGradient(miniBatch);

		for (Map.Entry<Neuron, List<Double>> entry : miniGradient.entrySet()) {
			Neuron neuron = entry.getKey();
			List<Double> gradientWeights = entry.getValue();
			List<Double> oldWeights = neuron.getWeights();

			List<Double> newWeights = Utils.zipLists(oldWeights, gradientWeights, (old, gradient) -> old-learningRate(time)*gradient);

			neuron.setWeights(newWeights);
		}

	}

	private double learningRate(double time) {

		return 0.05/(time/5+1);

	}


	public double error(Map<List<Double>, List<Double>> trainingSet) {

		double error = 0;

		for (Map.Entry<List<Double>, List<Double>> trainingSample : trainingSet.entrySet()) {
			List<Double> sampleInput = trainingSample.getKey();
			List<Double> desireOutput = trainingSample.getValue();

			List<Double> sampleOutput = computeOutput(sampleInput);

			List<Double> errorPerOutputs = Utils.zipLists(sampleOutput, desireOutput, (sample, desire) -> Math.pow(sample - desire, 2));

			for (double errorPerOutput : errorPerOutputs) {
				error += errorPerOutput;
			}

		}

		return error/2;

	}


	/**
	 * Computes gradient for each weight
	 *
	 * @param trainingSet
	 * @return
	 */
	private Map<Neuron, List<Double>> computeGradient(Map<List<Double>, List<Double>> trainingSet) {

		Map<Neuron, List<Double>> gradient = new HashMap<>();

		for (Map.Entry<List<Double>, List<Double>> trainingSample : trainingSet.entrySet()) {
			List<Double> sampleInput = trainingSample.getKey();
			List<Double> desireOutput = trainingSample.getValue();

			Map<Neuron, Double> neuronOutputs = forwardpass(sampleInput);

			Map<Neuron, Double> neuronGradients = backpropagation(sampleInput, desireOutput, neuronOutputs);


			for (Layer layer : layers) {
				Layer layerBelow = getLayerBelow(layer);

				for (Neuron neuron : layer.getNeurons()) {

					gradient.putIfAbsent(neuron, Utils.listOfZeros(neuron.getInputSize()));

					List<Double> weightGradientsPerNeuron = gradient.get(neuron);
					for (int i = 0; i < neuron.getWeights().size(); i++) {

						double yBelow;
						if (layerBelow == null) {
							yBelow = sampleInput.get(i);
						} else {
							Neuron neuronBelow = layerBelow.getNeurons().get(i);
							yBelow = neuronOutputs.get(neuronBelow);
						}

						double y = neuronOutputs.get(neuron);
						double g = neuronGradients.get(neuron);
						double d = neuron.derivationOutput(y);

						double weightGradient = g * d * yBelow;

						weightGradientsPerNeuron.set(i, weightGradientsPerNeuron.get(i) + weightGradient);

					}
					gradient.put(neuron, weightGradientsPerNeuron);

				}

			}

		}

		return gradient;

	}

	/**
	 * Compute output value for each neuron
	 *
	 * @param sampleInput
	 * @return
	 */
	private Map<Neuron, Double> forwardpass(List<Double> sampleInput) {

		Map<Neuron, Double> neuronsOutputs = new HashMap<>();

		List<Double> inputsForNextLayer = new ArrayList<>(sampleInput);

		for (Layer currentLayer : layers) {
			inputsForNextLayer = currentLayer.computeOutput(inputsForNextLayer);

			neuronsOutputs.putAll(Utils.mergeLists(currentLayer.getNeurons(), inputsForNextLayer));
		}
		return neuronsOutputs;

	}

	/**
	 * Compute gradient for each neuron
	 *
	 * @param sampleInput
	 * @param desireOutput
	 * @return
	 */
	private Map<Neuron, Double> backpropagation(List<Double> sampleInput, List<Double> desireOutput, Map<Neuron, Double> neuronOutputs) {
		List<Double> sampleOutput = computeOutput(sampleInput);

		// Set of all unstructured neurons of network. Result of this function.
		Map<Neuron, Double> neuronGradients = new HashMap<>();

		List<Layer> reverseLayers = new ArrayList<>(layers);
		Collections.reverse(reverseLayers);

		for (Layer layer : reverseLayers) {
			if (isLastLayer(layer)) {
				List<Double> aboveGradients = Utils.zipLists(sampleOutput, desireOutput, (sample, desire) -> sample - desire);
				neuronGradients.putAll(Utils.mergeLists(layer.getNeurons(), aboveGradients));
				continue;
			}

			Layer layerAbove = getLayerAbove(layer);

			for (Neuron neuron : layer.getNeurons()) {

				neuronGradients.put(neuron, 0.0);
				for (Neuron neuronAbove : layerAbove.getNeurons()) {

					double y = neuronOutputs.get(neuronAbove);
					double d = neuronAbove.derivationOutput(y);
					double w = neuronAbove.getWeights().get(layer.getNeuronPosition(neuron));
					double g = neuronGradients.get(neuronAbove);
					// sum partial results
					neuronGradients.put(neuron, neuronGradients.get(neuron) + g*d*w);

				}
			}
		}

		return neuronGradients;

	}


	private Layer getLayerBelow(Layer layer) {
		for (int i = 1; i < layers.size(); i++) {
			if (layers.get(i).equals(layer)) {
				return layers.get(i-1);
			}
		}
		return null;
	}

	private Layer getLayerAbove(Layer layer) {
		for (int i = 0; i < layers.size()-1; i++) {
			if (layers.get(i).equals(layer)) {
				return layers.get(i+1);
			}
		}
		return null;
	}

	private boolean isLastLayer(Layer layer) {
		return (layer.equals(layers.get(layers.size() - 1)));
	}



	public int getInputSize() {
		return layers.get(0).getInputSize();
	}

	public int getOutputSize() {
		return layers.get(layers.size() - 1).getOutputSize();
	}




	private void log(Object msg) {
		System.out.println(msg);
	}
}
