package cz.muni.fi.neural;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Created by ondrejvelisek on 11.11.16.
 */
public class Utils {

	public static <X, Y> Map<X, Y> mergeLists(List<X> keys, List<Y> values) {
		if (keys.size() != values.size()) {
			throw new IllegalArgumentException("Both lists has to have same size. " +
					"Keys size is "+keys.size()+", values size is "+values.size());
		}

		Map<X, Y> map = new HashMap<>();

		for (int i=0; i < keys.size(); i++) {
			map.put(keys.get(i), values.get(i));
		}

		return map;
	}

	public static <X, Y, Z> List<Z> zipLists(List<X> keys, List<Y> values, BiFunction<X, Y, Z> funtion) {
		List<Z> result = new ArrayList<>();

		for (Map.Entry<X, Y> entry : mergeLists(keys, values).entrySet()) {
			result.add(funtion.apply(entry.getKey(), entry.getValue()));
		}

		return result;
	}

	public static List<Double> listOfZeros(int size) {
		List<Double> result = new ArrayList<>();

		for (int i = 0; i < size; i++) {
			result.add(0.0);
		}

		return result;
	}

}
