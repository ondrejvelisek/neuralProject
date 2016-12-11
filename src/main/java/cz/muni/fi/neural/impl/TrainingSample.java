package cz.muni.fi.neural.impl;

import java.util.List;

/**
 * Created by ondrejvelisek on 4.12.16.
 */
public class TrainingSample {

	private List<Double> input;
	private List<Double> desireOutput;

	public TrainingSample(List<Double> input, List<Double> desireOutput) {
		this.input = input;
		this.desireOutput = desireOutput;
	}

	public List<Double> getInput() {
		return input;
	}

	public List<Double> getDesireOutput() {
		return desireOutput;
	}

}
