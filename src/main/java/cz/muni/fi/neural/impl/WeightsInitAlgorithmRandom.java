package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.lib.Weight;
import cz.muni.fi.neural.lib.WeightsInitAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by ondrejvelisek on 4.12.16.
 */
public class WeightsInitAlgorithmRandom implements WeightsInitAlgorithm {

	private double min;
	private double max;
	private Random random;

	public WeightsInitAlgorithmRandom(double min, double max) {
		this.min = min;
		this.max = max;
		this.random = new Random();
	}

	@Override
	public List<Weight> initWeights(int size) {
		List<Weight> list = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			list.add(new WeightImpl(min + max * random.nextDouble()));
		}
		return list;
	}

}
