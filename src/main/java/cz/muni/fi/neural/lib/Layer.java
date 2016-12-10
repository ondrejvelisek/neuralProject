package cz.muni.fi.neural.lib;

import java.util.List;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface Layer {

	List<Double> computeOutput(List<Double> inputs);

	int getInputSize();

	int getOutputSize();

	List<Neuron> getNeurons();

	int getNeuronPosition(Neuron neuron);
}
