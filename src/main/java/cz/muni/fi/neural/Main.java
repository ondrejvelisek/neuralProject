package cz.muni.fi.neural;

import java.io.*;
import java.util.*;
import java.util.logging.*;

public class Main {
    public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    public static void main(String[] args){
        ConfigReader  mlpConfig = ConfigReader.getInstance();
        DataReader dataReader = new DataReader();
        List<List<Double>> dataMatrix = null;

        try {
            LogManager.getLogManager().readConfiguration(new FileInputStream("src/main/resources/logger.properties"));
            FileHandler logFile = new FileHandler("./debug.log");
            Logger.getLogger(Logger.GLOBAL_LOGGER_NAME).addHandler(logFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            dataReader.setFile("xor_dataset.csv");
            dataMatrix = dataReader.csvToMatrix();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Creating network...");

        List<Integer> layersStructure = new ArrayList<>();
        int inputLayerSize = mlpConfig.getInputVectors().size();
        int outputLayerSize = 1;

        layersStructure.add(inputLayerSize);
        layersStructure.addAll(mlpConfig.getMlpArchitecture());
        layersStructure.add(outputLayerSize);

        NeuralNetwork net = new MultilayerPerceptron(layersStructure);

        // need to add random number otherwise map will 'melt' same rows. Create Object Inputs?
		Random rand = new Random();
		Map<List<Double>, List<Double>> trainingSet = new HashMap<>();
		for (List<Double> row : dataMatrix) {
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
