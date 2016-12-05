package cz.muni.fi.neural;

import java.text.DecimalFormat;
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
		for (int i = 1; i < numOfLayers - 1; i++) {
			int neuronsInLayer = layersStructure.get(i) + 1;
			int neuronsInPrevLayer = layersStructure.get(i-1) + 1;
			layers.add(constructLayer(neuronsInPrevLayer, neuronsInLayer));
		}
		int neuronsInOutputLayer = layersStructure.get(numOfLayers - 1);
		int neuronsInPrevLayer = layersStructure.get(numOfLayers - 2) + 1;
		layers.add(constructOutputLayer(neuronsInPrevLayer, neuronsInOutputLayer));


		if(ConfigReader.getInstance().initializationDebug()){
			logger.info("--------------------");
			logger.info("Multilayer perceptron builded: ");
			logger.info("Structure of MLP with bias neurons and without input layer: ");
			for (int i = 0; i < layers.size(); i++) {
				logger.info(""+(layers.get(i).getNeurons().size()));
			}
			logger.info("--------------------");
		}
	}

	private Layer constructLayer(int prevLayerSize, int layerSize) {

		List<Neuron> neurons = new ArrayList<>();

		for (int i = 0; i < layerSize - 1; i++) {
			neurons.add(new NeuronImpl(prevLayerSize));
		}
		neurons.add(new NeuronImpl()); //Bias neuron
		return new LayerImpl(neurons);
	}

	private Layer constructOutputLayer(int prevLayerSize, int layerSize) {

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

	public void learn(Double[][] inputsMatrix, Double[] outputsVector) {
		int trainingSetSize = outputsVector.length;

		ConfigReader  mlpConfig = ConfigReader.getInstance();
		int miniBatchSize = mlpConfig.getBatchSize();

		Double[][] miniBatchInputs = new Double[miniBatchSize][inputsMatrix[0].length];
		Double[] miniBatchOutputs = new Double[miniBatchSize];

		int t = 0;
		int currentSize = 0;
		if(mlpConfig.learningIterationsDebug()){
			logger.info("-------------LEARNING-------------");
			logger.info("Learning properties:");
			logger.info("Mini batch size: " + miniBatchSize);
			logger.info("----------------------------------");
		}
		for(int i = 0; i < 500; i++){
			for (int j = 0; j < trainingSetSize; j++) {
				miniBatchInputs[currentSize] = inputsMatrix[j];
				miniBatchOutputs[currentSize] = outputsVector[j];
				currentSize++;
				if (currentSize == miniBatchSize) {
					if(mlpConfig.learningIterationsDebug())
						logger.info("Iteration: " + t);

					miniLearn(t, miniBatchInputs, miniBatchOutputs);

					miniBatchInputs = new Double[miniBatchSize][inputsMatrix[0].length];
					miniBatchOutputs = new Double[miniBatchSize];
					currentSize = 0;
					t++;

					if(mlpConfig.learningIterationsDebug())
						logger.info("------------------------");
				}
			}
		}
	}

	private void miniLearn(int time, Double[][] miniBatchInputs, Double[] miniBatchOutputs) {

		Map<Neuron, List<Double>> miniGradient = computeGradient(miniBatchInputs,miniBatchOutputs);

		for (Map.Entry<Neuron, List<Double>> entry : miniGradient.entrySet()) {
			Neuron neuron = entry.getKey();
			List<Double> gradientWeights = entry.getValue();
			List<Double> oldWeights = neuron.getWeights();

			List<Double> newWeights = Utils.zipLists(oldWeights, gradientWeights, (old, gradient) -> old-learningRate(time)*gradient);

			neuron.setWeights(newWeights);
		}

	}

	private double learningRate(double time) {

		//return 0.05/(time/5+1);
		return 0.3;

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
	 * @return
	 */
	private Map<Neuron, List<Double>> computeGradient(Double[][] inputs, Double[] outputs) {
		int trainingSetSize = outputs.length;

		Map<Neuron, List<Double>> gradient = new HashMap<>();

		Double error = 0.0;
		for (int i = 0; i < trainingSetSize; i++){
			List<Double> sampleInput = new ArrayList<Double>(Arrays.asList(inputs[i]));
			List<Double> desireOutput = new ArrayList<Double>(Arrays.asList(outputs[i]));

			Map<Neuron, Double> neuronOutputs = forwardPass(sampleInput);

			Neuron outputNeuron = layers.get(layers.size()-1).getNeurons().get(0);
			List<Double> sampleOutput = new ArrayList<Double>(Arrays.asList(neuronOutputs.get(outputNeuron)));

			if(ConfigReader.getInstance().outputsOfLearningDebug()) {
				logger.info("output: " + new DecimalFormat("#0.00").format(sampleOutput.get(0)));
				logger.info("desire: " + desireOutput.get(0));
			}

			error += Math.pow(sampleOutput.get(0) - desireOutput.get(0), 2);

			Map<Neuron, Double> neuronGradients = backpropagation(sampleOutput, desireOutput, neuronOutputs);


			for (Layer layer : layers) {
				Layer layerBelow = getLayerBelow(layer);

				for (Neuron neuron : layer.getNeurons()) { //treba vynechat bias
					if(neuron.isBias()) continue;

					gradient.putIfAbsent(neuron, Utils.listOfZeros(neuron.getInputSize()));

					List<Double> weightGradientsPerNeuron = gradient.get(neuron);
					for (int j = 0; j < neuron.getWeights().size(); j++) {

						double yBelow;
						if (layerBelow == null) {
							yBelow = sampleInput.get(j);
						} else {
							Neuron neuronBelow = layerBelow.getNeurons().get(j);
							yBelow = neuronOutputs.get(neuronBelow);
						}

						double y = neuronOutputs.get(neuron);
						double g = neuronGradients.get(neuron);
						double d = neuron.derivationOutput(y);

						double weightGradient = g * d * yBelow;

						weightGradientsPerNeuron.set(j, weightGradientsPerNeuron.get(j) + weightGradient);

					}
					gradient.put(neuron, weightGradientsPerNeuron);
				}
			}
		}
		if(ConfigReader.getInstance().learningIterationsDebug())
			logger.info("Error MSE:"+error / trainingSetSize);
		return gradient;

	}


	/**
	 * Compute output value for each neuron
	 *
	 * @param sampleInput
	 * @return
	 */
	private Map<Neuron, Double> forwardPass(List<Double> sampleInput) {

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
	 * @param desireOutput
	 * @return
	 */
	private Map<Neuron, Double> backpropagation(List<Double>  sampleOutput, List<Double> desireOutput, Map<Neuron, Double> neuronOutputs) {

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
			//List<Neuron> neuronsOfLayer = layer.getNeurons();
			for (Neuron neuron : layer.getNeurons()) {

				neuronGradients.put(neuron, 0.0);
				for (Neuron neuronAbove : layerAbove.getNeurons()) {
					if(neuronAbove.isBias()) continue;

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
}
