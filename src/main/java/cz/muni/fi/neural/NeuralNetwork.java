package cz.muni.fi.neural;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface NeuralNetwork {

	List<Double> compute(List<Double> input);

	void learn(Map<List<Double>, List<Double>> trainingSet);

	int getInputSize();

	int getOutputSize();

}
