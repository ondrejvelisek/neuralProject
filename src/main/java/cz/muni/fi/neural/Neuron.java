package cz.muni.fi.neural;

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

	double derivationOutput(double y);

	List<Double> getWeights();

	void setWeights(List<Double> weights);

	int getInputSize();

	boolean isBias();

}
