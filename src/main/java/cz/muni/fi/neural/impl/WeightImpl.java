package cz.muni.fi.neural.impl;

import cz.muni.fi.neural.lib.Weight;

/**
 * Created by ondrejvelisek on 4.12.16.
 */
public class WeightImpl implements Weight {

	private double value;

	public WeightImpl(double value) {
		this.value = value;
	}

	@Override
	public double getValue() {
		return value;
	}

	@Override
	public void setValue(double value) {
		this.value = value;
	}
}

