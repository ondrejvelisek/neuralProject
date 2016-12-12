package cz.muni.fi.neural.lib;

import cz.muni.fi.neural.impl.DataSample;

import java.util.List;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface NeuralNetwork {

	List<Double> computeOutput(List<Double> input);

	void train(List<DataSample> trainingSet);

	int getInputSize();

	int getOutputSize();

	double error(List<DataSample> trainingSet);

}
