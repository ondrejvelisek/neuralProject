package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.ConfigReader;
import cz.muni.fi.neural.lib.*;
import cz.muni.fi.neural.Utils;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class MultilayerPerceptron implements NeuralNetwork {
	public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

	private List<Layer> layers;

	public MultilayerPerceptron(List<Integer> layersStructure, ActivationFunction ac, WeightsInitAlgorithm wia) {
		if (layersStructure.size() < 3) {
			throw new IllegalArgumentException("At least one hidden layer have to be present in MPL.");
		}

		layers = new ArrayList<>();
		for (int i = 1; i < layersStructure.size(); i++) {
			layers.add(new LayerImpl(layersStructure.get(i-1), layersStructure.get(i), ac, wia));
		}

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

	public List<Double> computeOutput(List<Double> inputs) {
		Utils.checkEqualSize(getInputSize(), inputs);

		List<Double> inputsForNextLayer = new ArrayList<>(inputs);

		for (Layer currentLayer : layers) {
			inputsForNextLayer = currentLayer.computeOutput(inputsForNextLayer);
		}
		return inputsForNextLayer;
	}

	@Override
	public void train(List<TrainingSample> trainingSet) {

		ConfigReader  mlpConfig = ConfigReader.getInstance();
		int miniBatchSize = mlpConfig.getBatchSize();

		int t = 0;
		List<TrainingSample> miniTrainingSet = new ArrayList<>();
		if(mlpConfig.learningIterationsDebug()){
			logger.info("-------------LEARNING-------------");
			logger.info("Learning properties:");
			logger.info("Mini batch size: " + miniBatchSize);
			logger.info("----------------------------------");
		}
		for(int i = 0; i < 50; i++){
			for (TrainingSample sample : trainingSet) {
				miniTrainingSet.add(sample);
				if (miniTrainingSet.size() == miniBatchSize) {
					if(mlpConfig.learningIterationsDebug())
						logger.info("Iteration: " + t);

					miniTrain(t, miniTrainingSet);

					miniTrainingSet.clear();
					t++;

					if(mlpConfig.learningIterationsDebug())
						logger.info("------------------------");
				}
			}
		}
	}

	private void miniTrain(int time, List<TrainingSample> miniTrainingSet) {

		Map<Weight, Double> miniGradient = computeGradient(miniTrainingSet);

		miniGradient.entrySet().forEach(entry -> entry.getKey().setValue(entry.getKey().getValue() - learningRate(time) * entry.getValue()));

	}

	private double learningRate(double time) {

		//return 0.05/(time/5+1);
		return 0.1;

	}

	/**
	 * Computes gradient for each weight
	 *
	 * @return
	 */
	private Map<Weight, Double> computeGradient(List<TrainingSample> trainingSet) {

		Map<Weight, Double> gradient = new HashMap<>();

		for (TrainingSample sample : trainingSet){

			Map<Neuron, Double> neuronOutputs = forwardPass(sample.getInput());

			//List<Neuron> outputNeurons = layers.get(layers.size()-1).getNeurons();
			List<Double> sampleOutput = computeOutput(sample.getInput());

			/*outputNeurons.stream()
					.filter(neuronOutputs::containsKey)
					.map(neuronOutputs::get)
					.collect(Collectors.toList());
			*/

			if(ConfigReader.getInstance().outputsOfLearningDebug()) {
				logger.info("input: " + sample.getInput());
				logger.info("output: " + sampleOutput);

				logger.info("desire: " + sample.getDesireOutput());
			}

			Map<Neuron, Double> neuronGradients = backpropagation(sampleOutput, sample.getDesireOutput(), neuronOutputs);


			for (Layer layer : layers) {

				for (Neuron neuron : layer.getNeurons()) {

					for (int w = 0; w < neuron.getWeights().size(); w++) {
						Weight weight = neuron.getWeights().get(w);

						double yBelow;
						if (w == neuron.getWeights().size()-1) {
							yBelow = 1.0;
						} else if (isFirstLayer(layer)) {
							yBelow = sample.getInput().get(w);
						} else {
							yBelow = neuronOutputs.get(getLayerBelow(layer).getNeurons().get(w));
						}

						double y = neuronOutputs.get(neuron);
						double g = neuronGradients.get(neuron);
						double d = neuron.derivationOutput(y);

						double weightGradient = g * d * yBelow;

						gradient.putIfAbsent(weight, 0.0);
						gradient.put(weight, gradient.get(weight) + weightGradient);

					}
				}
			}
		}
		if(ConfigReader.getInstance().learningIterationsDebug()) {
			logger.info("Error MSE:" + error(trainingSet));
		}

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

			for (Neuron neuron : layer.getNeurons()) {

				neuronGradients.put(neuron, 0.0);
				for (Neuron neuronAbove : layerAbove.getNeurons()) {

					double y = neuronOutputs.get(neuronAbove);
					double d = neuronAbove.derivationOutput(y);
					double w = neuronAbove.getWeights().get(layer.getNeuronPosition(neuron)).getValue();
					double g = neuronGradients.get(neuronAbove);
					// sum partial results
					neuronGradients.put(neuron, neuronGradients.get(neuron) + g*d*w);
				}
			}
		}
		return neuronGradients;
	}



	@Override
	public double error(List<TrainingSample> trainingSet) {

		double error = 0;

		for (TrainingSample sample : trainingSet) {

			List<Double> sampleOutput = computeOutput(sample.getInput());

			Double errorPerOutput = IntStream.range(0, sampleOutput.size())
					.mapToDouble(i -> Math.pow(sampleOutput.get(i) - sample.getDesireOutput().get(i), 2))
					.sum();

			error += errorPerOutput;

		}

		return error/trainingSet.size();

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

	private boolean isFirstLayer(Layer layer) {
		return (layer.equals(layers.get(0)));
	}


	public int getInputSize() {
		return layers.get(0).getInputSize();
	}

	public int getOutputSize() {
		return layers.get(layers.size() - 1).getOutputSize();
	}

	@Override
	public String toString() {
		return "MultilayerPerceptron{" +
				"\nlayers=" + layers +
				"\n}";
	}
}
