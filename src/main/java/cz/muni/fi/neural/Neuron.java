package cz.muni.fi.neural;

import java.util.List;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public abstract class Neuron implements NeuralNetwork {

	public int getOutputSize() {
		return 1;
	}

}
