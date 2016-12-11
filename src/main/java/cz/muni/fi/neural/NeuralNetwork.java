package cz.muni.fi.neural;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface NeuralNetwork {

	List<Double> computeOutput(List<Double> input);

	void learn(Double[][]trainingInputsMatrix, Double[] trainingOutputsVector, Double[][] validationInputsMatrix, Double[] validationOutputsVector);

	int getInputSize();

	int getOutputSize();

	double error(Double[][] input, Double[] output);

}
