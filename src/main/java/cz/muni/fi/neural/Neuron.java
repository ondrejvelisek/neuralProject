package cz.muni.fi.neural;

import java.util.List;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public abstract class Neuron implements NeuralNetwork {


	public int getOutputSize() {
		return 1;
	}

	public abstract double getLastOutput();
	public abstract double getLastDerivatedOutput();

	public abstract List<Double> getWeights();

	public abstract double getGradient();
	public abstract void setGradient(double gradient);

}
