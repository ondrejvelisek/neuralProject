package cz.muni.fi.neural.lib;

import java.util.List;

/**
 * Created by Simon on 06.11.2016.
 */
public interface WeightsInitAlgorithm {

	List<Weight> initWeights(int size);

}
