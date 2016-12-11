package cz.muni.fi.neural;

import cz.muni.fi.neural.impl.ActivationFunctionTanh;
import cz.muni.fi.neural.impl.MultilayerPerceptron;
import cz.muni.fi.neural.impl.TrainingSample;
import cz.muni.fi.neural.impl.WeightsInitAlgorithmRandom;
import cz.muni.fi.neural.lib.ActivationFunction;
import cz.muni.fi.neural.lib.NeuralNetwork;
import cz.muni.fi.neural.lib.WeightsInitAlgorithm;

import java.io.*;
import java.util.*;
import java.util.logging.*;

public class Main {
    public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    public static void main(String[] args){
        ConfigReader mlpConfig = ConfigReader.getInstance();
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

        dataMatrix = dataReader.normalize(dataMatrix);
        List<TrainingSample> trainingSet = dataReader.getTrainingSet(dataMatrix);

        List<Integer> layersStructure = new ArrayList<>();

        layersStructure.add(mlpConfig.getInputSize());
        layersStructure.addAll(mlpConfig.getMlpArchitecture());
        layersStructure.add(mlpConfig.getOutputSize());

        ActivationFunction ac = new ActivationFunctionTanh(1);
        WeightsInitAlgorithm wia = new WeightsInitAlgorithmRandom(-0.3, 0.3);

        int n = 100;
        int correct = 0;
        for (int i = 0; i < n; i++) {
            System.out.print(".");
        }
        System.out.println();
        for (int i = 0; i < n; i++) {

            NeuralNetwork net = new MultilayerPerceptron(layersStructure, ac, wia);

            double origErr = net.error(trainingSet);

            net.train(trainingSet);

            double trainedError = net.error(trainingSet);
            if (trainedError < origErr/10) {
                correct++;
            }
            System.out.print("|");
        }
        System.out.println();
        System.out.println("Total experiments: " + n);
        System.out.println("Successful experiments: " + correct);


    }

}
