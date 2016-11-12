package cz.muni.fi.neural;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class Main {

    public static void main(String[] args) throws IOException {

        System.out.println("Creating network...");
        NeuralNetwork net = new MultilayerPerceptron(2, 8, 1);

        DataReader dataReader = new DataReader();
        ArrayList<ArrayList<Double>> matrix = null;
        try {
            dataReader.setFile("xor_dataset.csv");
            matrix = dataReader.csvToMatrix();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // need to add random number otherwise map will 'melt' same rows. Create Object Inputs?
		Random rand = new Random();
		Map<List<Double>, List<Double>> trainingSet = new HashMap<>();
		for (List<Double> row : matrix) {
			trainingSet.put(Arrays.asList(row.get(0), row.get(1), rand.nextDouble()), Collections.singletonList(row.get(2)));
		}

		double error = net.error(trainingSet);
		System.out.println(error);

		System.out.println("trainingSet size: " + trainingSet.size());

		net.learn(trainingSet);

		System.out.println();
		System.out.println(error - net.error(trainingSet));


    }

}
