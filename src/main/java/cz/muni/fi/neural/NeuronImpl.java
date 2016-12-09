package cz.muni.fi.neural;

import cz.muni.fi.neural.algorithms.ActivationFunction;
import cz.muni.fi.neural.algorithms.ActivationFunctionTanh;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public class NeuronImpl implements Neuron {
	public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME); //negetovat ho vsade

	private boolean biasNeuron;
	private List<Double> weights;
	private ActivationFunction activationFunction;

	public NeuronImpl(int inputSize) {
		this.weights = generateWeights(inputSize);
		activationFunction = new ActivationFunctionTanh();
		this.biasNeuron = false;
	}

	public NeuronImpl() {
		this.biasNeuron = true;
	}

	private List<Double> generateWeights(int inputSize) {
		Random r = new Random();
		List<Double> list = new ArrayList<>();
		for (int i = 0; i < inputSize; i++) {
			list.add(-0.1 + 0.2 * r.nextDouble());
			//list.add(0.1);
		}
		return list;

	}

	public double computeOutput(List<Double> inputs) {
		if (inputs.size() != weights.size()) {
			throw new IllegalArgumentException();
		}

		double innerPotential = computeInnerPotential(inputs);
		return activationFunction.computeOutput(innerPotential);
	}

	private double computeInnerPotential(List<Double> inputs) {

		double sum = 0;
		for(int i = 0; i < weights.size(); i++) {
			if(ConfigReader.getInstance().neuronInputsDebug()){
				logger.info(""+inputs.get(i));
			}
			sum += inputs.get(i) * weights.get(i);
		}

		if(ConfigReader.getInstance().neuronInputsDebug()){
			logger.info("--------------------------");
		}

		return sum;

	}

	public int getInputSize() {
		if(weights == null)
			return 0;

		return weights.size();
	}

	public double derivationOutput(double y) {
		return activationFunction.derivationOutput(y);
	}

	public List<Double> getWeights() {
		return weights;
	}

	@Override
	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}

	public boolean isBias(){
		return biasNeuron;
	}

}
