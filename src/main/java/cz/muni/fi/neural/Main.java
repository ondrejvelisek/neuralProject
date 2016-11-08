package cz.muni.fi.neural;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static void main(String[] args) throws IOException {

        System.out.println("Creating network...");
        NeuralNetwork net = new MultilayerPerceptron(2, 16, 64, 16, 1);

        DataReader dataReader = new DataReader();
        ArrayList<ArrayList<Double>> matrix = null;
        try {
            dataReader.setFile("xor_dataset.csv");
            matrix = dataReader.csvToMatrix();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        for(int i=0; i < matrix.size(); i++) {
            System.out.print(i + " ");
            for (int j = 0; j < matrix.get(i).size(); j++) {
                System.out.print(matrix.get(i).get(j) + " ");
            }
            System.out.println();
        }
    }

}
