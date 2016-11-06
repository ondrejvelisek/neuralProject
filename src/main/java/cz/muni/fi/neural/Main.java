package cz.muni.fi.neural;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {

    public static void main(String[] args) {

        List<Double> input0 = Arrays.asList(0.00, 0.01);
        List<Double> input1 = Arrays.asList(0.01, 0.02);
        List<Double> input2 = Arrays.asList(0.00, 0.00);
        List<Double> input3 = Arrays.asList(0.03, 0.02);

        System.out.println("Creating network...");
        NeuralNetwork net = new MultilayerPerceptron(2, 16, 64, 16, 1);

        System.out.println("Computing sum of "+input0);
        List<Double> output = net.computeOutput(input0);

        System.out.println(output);

        System.out.println("Learning (sum of four numbers)...");

        Map<List<Double>, List<Double>> trainingSet = new HashMap<>();

        trainingSet.put(Arrays.asList(0.00, 0.00), Arrays.asList(0.00));
        trainingSet.put(Arrays.asList(0.00, 0.01), Arrays.asList(0.01));
        trainingSet.put(Arrays.asList(0.00, 0.02), Arrays.asList(0.02));
        trainingSet.put(Arrays.asList(0.00, 0.03), Arrays.asList(0.03));

        trainingSet.put(Arrays.asList(0.01, 0.00), Arrays.asList(0.01));
        trainingSet.put(Arrays.asList(0.01, 0.01), Arrays.asList(0.02));
        trainingSet.put(Arrays.asList(0.01, 0.02), Arrays.asList(0.03));
        trainingSet.put(Arrays.asList(0.01, 0.03), Arrays.asList(0.04));

        trainingSet.put(Arrays.asList(0.02, 0.00), Arrays.asList(0.02));
        trainingSet.put(Arrays.asList(0.02, 0.01), Arrays.asList(0.03));
        trainingSet.put(Arrays.asList(0.02, 0.02), Arrays.asList(0.04));
        trainingSet.put(Arrays.asList(0.02, 0.03), Arrays.asList(0.05));

        trainingSet.put(Arrays.asList(0.03, 0.00), Arrays.asList(0.03));
        trainingSet.put(Arrays.asList(0.03, 0.01), Arrays.asList(0.04));
        trainingSet.put(Arrays.asList(0.03, 0.02), Arrays.asList(0.05));
        trainingSet.put(Arrays.asList(0.03, 0.03), Arrays.asList(0.06));

        trainingSet.put(Arrays.asList(0.01, 0.00), Arrays.asList(0.01));
        trainingSet.put(Arrays.asList(0.02, 0.00), Arrays.asList(0.02));
        trainingSet.put(Arrays.asList(0.03, 0.00), Arrays.asList(0.03));

        trainingSet.put(Arrays.asList(0.00, 0.01), Arrays.asList(0.01));
        trainingSet.put(Arrays.asList(0.02, 0.01), Arrays.asList(0.03));
        trainingSet.put(Arrays.asList(0.03, 0.01), Arrays.asList(0.04));

        trainingSet.put(Arrays.asList(0.00, 0.02), Arrays.asList(0.02));
        trainingSet.put(Arrays.asList(0.01, 0.02), Arrays.asList(0.03));
        trainingSet.put(Arrays.asList(0.03, 0.02), Arrays.asList(0.05));

        trainingSet.put(Arrays.asList(0.00, 0.03), Arrays.asList(0.03));
        trainingSet.put(Arrays.asList(0.01, 0.03), Arrays.asList(0.04));
        trainingSet.put(Arrays.asList(0.02, 0.03), Arrays.asList(0.05));

        net.learn(trainingSet);
	    net.learn(trainingSet);
	    net.learn(trainingSet);

        System.out.println("Computing sum of "+input0);
        System.out.println(net.computeOutput(input0));
        System.out.println("Computing sum of "+input1);
        System.out.println(net.computeOutput(input1));
        System.out.println("Computing sum of "+input2);
        System.out.println(net.computeOutput(input2));
        System.out.println("Computing sum of "+input3);
        System.out.println(net.computeOutput(input3));

    }

}
