package cz.muni.fi.neural.lib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface Neuron {

	double computeOutput(List<Double> inputs);

	double derivationOutput(double output);

	List<Weight> getWeights();

	List<Double> getWeightValues();

	void updateWeights(List<Double> weights);

	int getInputSize();

}
