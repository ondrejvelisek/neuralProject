package cz.muni.fi.neural;

import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) {

        List<Double> input = Arrays.asList(1., 10., 23., 0.023, 12., -5., 12., -20.);

        System.out.println(input);

        System.out.println("Creating network...");
        NeuralNetwork net = new MultilayerPerceptron(8, 1024, 512, 256, 16);

        System.out.println("Computing...");
        List<Double> output = net.compute(input);

        System.out.println(output);
    }

}
