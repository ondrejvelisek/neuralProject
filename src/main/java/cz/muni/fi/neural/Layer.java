package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
