package cz.muni.fi.neural;

import java.util.List;
import java.util.Set;

/**
 * @author Ondrej Velisek <ondrejvelisek@gmail.com>
 */
public interface NeuralNetwork {

	List<Double> compute(List<Double> input);

	int getInputSize();

	int getOutputSize();

}
