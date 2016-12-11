package cz.muni.fi.neural.lib;

import cz.muni.fi.neural.impl.TrainingSample;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface NeuralNetwork {

	List<Double> computeOutput(List<Double> input);

	void train(List<TrainingSample> trainingSet);

	int getInputSize();

	int getOutputSize();

	double error(List<TrainingSample> trainingSet);

}
