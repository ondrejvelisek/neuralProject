package cz.muni.fi.neural;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface NeuralNetwork {

	List<Double> computeOutput(List<Double> input);

	void learn(Double[][]inputsMatrix, Double[] outputsVector);

	int getInputSize();

	int getOutputSize();

	double error(Map<List<Double>, List<Double>> trainingSet);

}
